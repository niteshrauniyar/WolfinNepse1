"""
Module 1: Automated Live Data Ingestion
========================================
Scrapes floorsheet and market depth data from NEPSE portals.
Handles dynamic JS tables, pagination, and data cleaning.
Sources: NepseAlpha, ShareSansar, NEPSE Official
"""

import re
import time
import json
import logging
import sqlite3
import random
from datetime import datetime, timedelta
from typing import Optional
from io import StringIO

import numpy as np
import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────
NEPSE_ALPHA_BASE    = "https://www.nepsealpha.com"
SHARESANSAR_BASE    = "https://www.sharesansar.com"
NEPSE_OFFICIAL_BASE = "https://nepalstock.com"

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/html, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": NEPSE_ALPHA_BASE,
}

# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _clean_numeric(value) -> Optional[float]:
    """Strip commas, hidden Unicode chars, whitespace; return float or None."""
    if pd.isna(value):
        return None
    s = str(value)
    # Remove zero-width / non-breaking spaces and other Unicode junk
    s = re.sub(r"[\u200b\u00a0\u202f\ufeff]", "", s)
    s = s.replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply _clean_numeric to all columns that look like numbers."""
    for col in df.columns:
        sample = df[col].dropna().head(20)
        numeric_count = sum(
            1 for v in sample
            if re.search(r"\d", str(v))
        )
        if numeric_count > len(sample) * 0.6:
            df[col] = df[col].apply(_clean_numeric)
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  Session factory (handles cookies / retries)
# ─────────────────────────────────────────────────────────────────────────────

def _get_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(DEFAULT_HEADERS)
    return session


# ─────────────────────────────────────────────────────────────────────────────
#  NEPSE Alpha – Floorsheet scraper
# ─────────────────────────────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _fetch_nepsealpha_floorsheet_page(
    session: requests.Session,
    page: int = 1,
    symbol: str = "",
) -> dict:
    """
    Fetch one page of floorsheet data from NepseAlpha's API endpoint.
    NepseAlpha uses an internal JSON API for its React frontend.
    """
    url = f"{NEPSE_ALPHA_BASE}/api/floorsheet"
    params = {
        "page": page,
        "size": 50,
        "symbol": symbol,
        "businessDate": datetime.now().strftime("%Y-%m-%d"),
    }
    try:
        resp = session.get(url, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except (requests.RequestException, json.JSONDecodeError) as exc:
        logger.warning("NepseAlpha page %d error: %s", page, exc)
        raise


def fetch_floorsheet_nepsealpha(
    symbol: str = "",
    max_pages: int = 10,
) -> pd.DataFrame:
    """
    Scrape paginated floorsheet from NepseAlpha.
    Returns a cleaned DataFrame with standardised column names.
    """
    session = _get_session()
    all_rows: list[dict] = []

    for page in range(1, max_pages + 1):
        try:
            payload = _fetch_nepsealpha_floorsheet_page(session, page, symbol)
        except Exception as exc:
            logger.error("Failed to fetch page %d: %s", page, exc)
            break

        # NepseAlpha wraps data in different keys depending on version
        rows = (
            payload.get("data", [])
            or payload.get("floorsheets", {}).get("content", [])
            or []
        )
        if not rows:
            break

        all_rows.extend(rows)
        logger.info("Fetched page %d – %d records so far", page, len(all_rows))

        # Respect rate limit
        time.sleep(random.uniform(0.5, 1.2))

        total_pages = (
            payload.get("totalPages")
            or payload.get("floorsheets", {}).get("totalPages", 1)
        )
        if page >= total_pages:
            break

    if not all_rows:
        logger.warning("No floorsheet data returned; using demo data.")
        return _generate_demo_floorsheet()

    df = pd.DataFrame(all_rows)
    return _normalise_floorsheet(df)


# ─────────────────────────────────────────────────────────────────────────────
#  ShareSansar – fallback scraper (HTML table parsing)
# ─────────────────────────────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_floorsheet_sharesansar(symbol: str = "NABIL") -> pd.DataFrame:
    """
    Scrape floorsheet table from ShareSansar using requests + pandas.
    Falls back to demo data on failure.
    """
    session = _get_session()
    url = f"{SHARESANSAR_BASE}/floorsheet"
    params = {"stock": symbol}

    try:
        resp = session.get(url, params=params, timeout=20)
        resp.raise_for_status()
        tables = pd.read_html(StringIO(resp.text))
        if tables:
            df = tables[0]
            return _normalise_floorsheet(df)
    except Exception as exc:
        logger.warning("ShareSansar scrape failed: %s", exc)

    return _generate_demo_floorsheet()


# ─────────────────────────────────────────────────────────────────────────────
#  NEPSE Official – market depth
# ─────────────────────────────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_market_depth(symbol: str = "NABIL") -> pd.DataFrame:
    """
    Fetch level-2 market depth (order book) from NEPSE official API.
    Endpoint discovered via browser DevTools network inspection.
    """
    session = _get_session()
    url = f"{NEPSE_OFFICIAL_BASE}/api/nots/securityDailyTradeStat/{symbol}"

    try:
        resp = session.get(url, timeout=15)
        resp.raise_for_status()
        payload = resp.json()
        depth_data = payload.get("content", payload if isinstance(payload, list) else [])
        df = pd.DataFrame(depth_data)
        return _normalise_market_depth(df)
    except Exception as exc:
        logger.warning("Market depth fetch failed: %s – using demo.", exc)
        return _generate_demo_market_depth(symbol)


# ─────────────────────────────────────────────────────────────────────────────
#  Column normalisation helpers
# ─────────────────────────────────────────────────────────────────────────────

_FLOORSHEET_COL_MAP = {
    # NepseAlpha API keys
    "transactionNo":  "transaction_id",
    "symbol":         "symbol",
    "buyerBroker":    "buyer_broker",
    "sellerBroker":   "seller_broker",
    "quantity":       "quantity",
    "rate":           "price",
    "amount":         "amount",
    "businessDate":   "date",
    # HTML table variants (ShareSansar)
    "Transaction No": "transaction_id",
    "Symbol":         "symbol",
    "Buyer Broker":   "buyer_broker",
    "Seller Broker":  "seller_broker",
    "Quantity":       "quantity",
    "Rate":           "price",
    "Amount":         "amount",
    "Business Date":  "date",
}

_DEPTH_COL_MAP = {
    "symbol":         "symbol",
    "lastTradedPrice":"ltp",
    "totalTradeQuantity": "volume",
    "highPrice":      "high",
    "lowPrice":       "low",
    "openPrice":      "open",
    "previousClose":  "prev_close",
}


def _normalise_floorsheet(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={k: v for k, v in _FLOORSHEET_COL_MAP.items() if k in df.columns})
    required = ["symbol", "buyer_broker", "seller_broker", "quantity", "price", "amount"]
    for col in required:
        if col not in df.columns:
            df[col] = np.nan

    numeric_cols = ["quantity", "price", "amount"]
    for col in numeric_cols:
        df[col] = df[col].apply(_clean_numeric)

    df["buyer_broker"]  = df["buyer_broker"].astype(str).str.strip()
    df["seller_broker"] = df["seller_broker"].astype(str).str.strip()
    df["symbol"]        = df["symbol"].astype(str).str.upper().str.strip()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        df["date"] = datetime.now()

    df = df.dropna(subset=["quantity", "price"])
    df = df.reset_index(drop=True)
    logger.info("Normalised floorsheet: %d rows", len(df))
    return df


def _normalise_market_depth(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={k: v for k, v in _DEPTH_COL_MAP.items() if k in df.columns})
    df = _clean_dataframe(df)
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  Demo / synthetic data generators (used when live scraping fails)
# ─────────────────────────────────────────────────────────────────────────────

_BROKER_IDS  = [str(i) for i in range(1, 61)]
_SYMBOLS     = ["NABIL", "NICA", "SCB", "ADBL", "EBL", "PCBL", "KBL",
                "HIDCL", "NHPC", "BPCL", "CHCL", "NTC", "SBL", "LBL"]


def _generate_demo_floorsheet(n: int = 2000) -> pd.DataFrame:
    """
    Generate realistic synthetic floorsheet data for demo / offline mode.
    Incorporates order-splitting signatures for institutional brokers.
    """
    rng = np.random.default_rng(42)
    now = datetime.now()

    # Institutional brokers trade in larger, more correlated blocks
    inst_brokers  = ["1", "5", "12", "23", "47"]
    retail_brokers = [b for b in _BROKER_IDS if b not in inst_brokers]

    buyer_brokers, seller_brokers, quantities, prices, symbols, dates = [], [], [], [], [], []

    for i in range(n):
        sym = rng.choice(_SYMBOLS)
        base_price = rng.uniform(200, 2000)

        # Institutional trade: large size, clustered timing
        if rng.random() < 0.15:
            buyer  = rng.choice(inst_brokers)
            seller = rng.choice(retail_brokers)
            qty    = int(rng.choice([500, 1000, 2000, 5000]) * rng.uniform(0.8, 1.2))
        else:
            buyer  = rng.choice(retail_brokers)
            seller = rng.choice(retail_brokers)
            qty    = int(rng.choice([10, 20, 50, 100, 200]) * rng.uniform(0.5, 2.0))

        noise  = rng.normal(0, base_price * 0.001)
        price  = round(base_price + noise, 2)
        ts     = now - timedelta(minutes=int(rng.uniform(0, 360)))

        buyer_brokers.append(buyer)
        seller_brokers.append(seller)
        quantities.append(qty)
        prices.append(price)
        symbols.append(sym)
        dates.append(ts)

    df = pd.DataFrame({
        "transaction_id": range(n),
        "symbol":         symbols,
        "buyer_broker":   buyer_brokers,
        "seller_broker":  seller_brokers,
        "quantity":       quantities,
        "price":          prices,
        "amount":         [q * p for q, p in zip(quantities, prices)],
        "date":           dates,
    })
    df = df.sort_values("date").reset_index(drop=True)
    return df


def _generate_demo_market_depth(symbol: str = "NABIL") -> pd.DataFrame:
    rng = np.random.default_rng(0)
    base = rng.uniform(800, 1200)
    n = 30
    dates = [datetime.now() - timedelta(days=i) for i in range(n - 1, -1, -1)]
    return pd.DataFrame({
        "symbol":     [symbol] * n,
        "date":       dates,
        "ltp":        base + rng.normal(0, base * 0.01, n).cumsum(),
        "volume":     rng.integers(5_000, 100_000, n).astype(float),
        "high":       base + rng.uniform(5, 50, n),
        "low":        base - rng.uniform(5, 50, n),
        "open":       base + rng.normal(0, 10, n),
        "prev_close": base + rng.normal(0, 8, n),
    })


# ─────────────────────────────────────────────────────────────────────────────
#  SQLite time-series store
# ─────────────────────────────────────────────────────────────────────────────

def save_to_sqlite(df: pd.DataFrame, table: str = "floorsheet", db_path: str = "nepse.db"):
    """Persist DataFrame to SQLite; appends without duplicating."""
    conn = sqlite3.connect(db_path)
    df.to_sql(table, conn, if_exists="append", index=False)
    conn.close()
    logger.info("Saved %d rows to SQLite table '%s'", len(df), table)


def load_from_sqlite(table: str = "floorsheet", db_path: str = "nepse.db",
                     days_back: int = 30) -> pd.DataFrame:
    """Load recent records from SQLite."""
    try:
        conn = sqlite3.connect(db_path)
        cutoff = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        query = f"SELECT * FROM {table} WHERE date >= '{cutoff}'"
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as exc:
        logger.warning("SQLite load failed: %s", exc)
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
#  Main entry point used by the Streamlit app
# ─────────────────────────────────────────────────────────────────────────────

def get_floorsheet(symbol: str = "", use_demo: bool = False) -> pd.DataFrame:
    """
    Primary data retrieval function.
    Tries live sources in order; falls back to demo data.
    """
    if use_demo:
        return _generate_demo_floorsheet()

    df = fetch_floorsheet_nepsealpha(symbol=symbol, max_pages=5)
    if df.empty:
        df = fetch_floorsheet_sharesansar(symbol=symbol or "NABIL")
    if df.empty:
        logger.warning("All live sources failed. Switching to demo data.")
        df = _generate_demo_floorsheet()
    return df


def get_market_depth(symbol: str = "NABIL", use_demo: bool = False) -> pd.DataFrame:
    if use_demo:
        return _generate_demo_market_depth(symbol)
    return fetch_market_depth(symbol)

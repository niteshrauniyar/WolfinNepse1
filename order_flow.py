"""
Module 2: Order Flow & Trade Size Engine
==========================================
Implements:
  - Order flow autocorrelation & metaorder detection (Tsaknaki et al. 2023)
  - Bayesian change-point detection via ruptures (Boehmer et al. 2020)
  - ADV-based institutional trade size filtering (top-5% flagging)
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd
import ruptures as rpt
from scipy.stats import pearsonr

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Trade sign classification (Lee-Ready proxy)
# ─────────────────────────────────────────────────────────────────────────────

def classify_trade_sign(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign +1 (buyer-initiated) or -1 (seller-initiated) to each trade
    using a simplified Lee-Ready rule on price movement.
    """
    df = df.sort_values("date").copy()
    df["price_change"] = df.groupby("symbol")["price"].diff()
    df["trade_sign"] = np.where(df["price_change"] > 0, 1,
                        np.where(df["price_change"] < 0, -1, 0))
    # Carry forward last non-zero sign for zero-change ticks
    df["trade_sign"] = df.groupby("symbol")["trade_sign"].transform(
        lambda s: s.replace(0, np.nan).ffill().fillna(1)
    ).astype(int)
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  Order Flow Imbalance (OFI)
# ─────────────────────────────────────────────────────────────────────────────

def compute_ofi(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Order Flow Imbalance = signed_volume / total_volume (rolling).
    High absolute OFI indicates persistent directional pressure.
    """
    df = df.sort_values("date").copy()
    df["signed_qty"] = df["quantity"] * df["trade_sign"]
    df["ofi"] = (
        df.groupby("symbol")["signed_qty"]
          .transform(lambda s: s.rolling(window, min_periods=1).sum())
        / df.groupby("symbol")["quantity"]
          .transform(lambda s: s.rolling(window, min_periods=1).sum())
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  Autocorrelation of order flow per broker (metaorder detection)
# ─────────────────────────────────────────────────────────────────────────────

def compute_broker_autocorrelation(
    df: pd.DataFrame,
    max_lag: int = 10,
) -> pd.DataFrame:
    """
    Tsaknaki et al. (2023): persistent autocorrelation in a broker's
    trade-sign sequence signals order splitting (metaorder execution).

    Returns a DataFrame with one row per broker:
      - mean_autocorr : average ACF across lags 1..max_lag
      - max_autocorr  : maximum ACF value
      - is_metaorder  : flag if mean_autocorr > threshold (0.15)
    """
    records = []
    for broker, grp in df.groupby("buyer_broker"):
        signs = grp.sort_values("date")["trade_sign"].values
        if len(signs) < max_lag + 5:
            continue

        acf_values = []
        for lag in range(1, max_lag + 1):
            if len(signs) > lag:
                try:
                    corr, _ = pearsonr(signs[:-lag], signs[lag:])
                    acf_values.append(corr if not np.isnan(corr) else 0.0)
                except Exception:
                    acf_values.append(0.0)

        if not acf_values:
            continue

        mean_acf = float(np.mean(acf_values))
        max_acf  = float(np.max(acf_values))
        n_trades = len(signs)

        records.append({
            "broker":          broker,
            "n_trades":        n_trades,
            "mean_autocorr":   round(mean_acf, 4),
            "max_autocorr":    round(max_acf, 4),
            "is_metaorder":    mean_acf > 0.15,
            "acf_series":      acf_values,
        })

    result = pd.DataFrame(records)
    if not result.empty:
        result = result.sort_values("mean_autocorr", ascending=False).reset_index(drop=True)
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Change-Point Detection (Bayesian / ruptures)
# ─────────────────────────────────────────────────────────────────────────────

def detect_changepoints(
    df: pd.DataFrame,
    symbol: str,
    signal_col: str = "ofi",
    model: str = "rbf",
    n_bkps: int = 5,
) -> Tuple[list[int], np.ndarray]:
    """
    Boehmer et al. (2020): detect structural breaks in order flow using
    ruptures' kernel change-point detection (PELT or Binseg).

    Returns:
        breakpoints : list of row indices where regime changes occur
        signal      : the signal array used
    """
    sub = df[df["symbol"] == symbol].sort_values("date").copy()
    if len(sub) < 20:
        return [], np.array([])

    if signal_col not in sub.columns:
        sub = compute_ofi(sub)

    signal = sub[signal_col].fillna(0).values.astype(float)

    try:
        algo = rpt.Pelt(model=model, min_size=5, jump=2).fit(signal)
        breakpoints = algo.predict(pen=3)        # penalty tuned for NEPSE tick frequency
        # ruptures returns the last index as endpoint; drop it
        breakpoints = [b for b in breakpoints if b < len(signal)]
    except Exception as exc:
        logger.warning("PELT failed (%s); falling back to Binseg.", exc)
        try:
            algo = rpt.Binseg(model="l2").fit(signal)
            breakpoints = algo.predict(n_bkps=n_bkps)
            breakpoints = [b for b in breakpoints if b < len(signal)]
        except Exception as exc2:
            logger.error("Change-point detection failed: %s", exc2)
            breakpoints = []

    logger.info("Symbol %s: %d changepoints detected", symbol, len(breakpoints))
    return breakpoints, signal


# ─────────────────────────────────────────────────────────────────────────────
#  ADV Trade Size Filter (top-5% institutional flag)
# ─────────────────────────────────────────────────────────────────────────────

def compute_adv_filter(df: pd.DataFrame, adv_window: int = 30) -> pd.DataFrame:
    """
    Flag trades in the top 5th percentile of quantity distribution per symbol
    as potentially institutional. Computes per-symbol 30-day ADV proxy.

    Adds columns:
        adv              : rolling average daily volume
        qty_pct_rank     : percentile rank of each trade's quantity
        is_institutional : True if qty_pct_rank >= 0.95
    """
    df = df.sort_values("date").copy()

    # Daily volume per symbol
    df["trade_date"] = df["date"].dt.date
    daily_vol = (
        df.groupby(["symbol", "trade_date"])["quantity"]
          .sum()
          .reset_index()
          .rename(columns={"quantity": "daily_volume"})
    )
    daily_vol["adv"] = (
        daily_vol.groupby("symbol")["daily_volume"]
                 .transform(lambda s: s.rolling(adv_window, min_periods=1).mean())
    )

    df = df.merge(
        daily_vol[["symbol", "trade_date", "adv"]],
        on=["symbol", "trade_date"],
        how="left",
    )

    # Per-symbol quantity percentile rank
    df["qty_pct_rank"] = df.groupby("symbol")["quantity"].transform(
        lambda s: s.rank(pct=True)
    )
    df["is_institutional"] = df["qty_pct_rank"] >= 0.95

    logger.info(
        "ADV filter: %d / %d trades flagged institutional (%.1f%%)",
        df["is_institutional"].sum(),
        len(df),
        df["is_institutional"].mean() * 100,
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  Broker-level order flow summary
# ─────────────────────────────────────────────────────────────────────────────

def broker_order_flow_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-broker buy/sell volumes, net position, and OFI score.
    Useful for downstream network and clustering modules.
    """
    buy_agg = (
        df.groupby("buyer_broker")
          .agg(buy_volume=("quantity", "sum"),
               buy_amount=("amount", "sum"),
               buy_trades=("quantity", "count"))
          .rename_axis("broker")
    )
    sell_agg = (
        df.groupby("seller_broker")
          .agg(sell_volume=("quantity", "sum"),
               sell_amount=("amount", "sum"),
               sell_trades=("quantity", "count"))
          .rename_axis("broker")
    )
    summary = buy_agg.join(sell_agg, how="outer").fillna(0)
    summary["net_volume"]   = summary["buy_volume"]  - summary["sell_volume"]
    summary["net_amount"]   = summary["buy_amount"]  - summary["sell_amount"]
    summary["total_volume"] = summary["buy_volume"]  + summary["sell_volume"]
    summary["total_trades"] = summary["buy_trades"]  + summary["sell_trades"]
    summary["ofi_score"]    = summary["net_volume"] / summary["total_volume"].replace(0, np.nan)

    # Avg trade size – proxy for institutional behaviour
    summary["avg_trade_size"] = summary["total_volume"] / summary["total_trades"].replace(0, np.nan)

    return summary.reset_index().sort_values("total_volume", ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
#  Convenience: run full Module 2 pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_order_flow_engine(df: pd.DataFrame) -> dict:
    """
    Run all Module 2 analytics. Returns a dict of result DataFrames / arrays.
    """
    df = classify_trade_sign(df)
    df = compute_ofi(df)
    df = compute_adv_filter(df)

    broker_acf      = compute_broker_autocorrelation(df)
    broker_summary  = broker_order_flow_summary(df)

    return {
        "enriched_df":    df,
        "broker_acf":     broker_acf,
        "broker_summary": broker_summary,
    }

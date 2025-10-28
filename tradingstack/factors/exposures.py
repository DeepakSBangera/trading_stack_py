from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ---------- sector mapping ----------
def load_sector_mapping(csv_path: Path | str) -> dict[str, str]:
    """
    CSV format:
        ticker,sector
        INFY.NS,Information Technology
        ...
    Returns dict[ticker] -> "sector_<normalized>"
    """
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"Sector mapping not found: {p}")
    df = pd.read_csv(p)
    cols = {c.strip().lower(): c for c in df.columns}
    if "ticker" not in cols or "sector" not in cols:
        raise ValueError("sector CSV must have columns: ticker, sector")
    tick = df[cols["ticker"]].astype(str).str.strip()
    sec = (
        df[cols["sector"]].astype(str).str.strip().str.replace(r"\s+", "_", regex=True)
    )
    return {t: f"sector_{s}" for t, s in zip(tick, sec, strict=False)}


# ---------- rolling sector exposure ----------
def rolling_sector_exposures_from_weights(
    weights: pd.DataFrame,
    mapping: dict[str, str],
    window: int = 63,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """
    Map ticker weights -> sector weights, then compute rolling mean exposures.
    `weights` must have columns: ['ticker','weight'] and a DatetimeIndex (daily).
    """
    minp = window if min_periods is None else int(min_periods)

    if not {"ticker", "weight"} <= set(weights.columns):
        raise ValueError("weights requires columns: ticker, weight")

    w = weights.copy()
    # normalize per day
    w["weight"] = pd.to_numeric(w["weight"], errors="coerce").fillna(0.0)
    w = w.pivot_table(
        index=w.index, columns="ticker", values="weight", aggfunc="sum"
    ).fillna(0.0)
    rowsum = w.abs().sum(axis=1).replace(0.0, np.nan)
    w = w.div(rowsum, axis=0).fillna(0.0)

    # map tickers -> sectors (unknown -> sector_Unknown)
    tickers = list(w.columns)
    sec_cols = [mapping.get(t, "sector_Unknown") for t in tickers]
    # collapse by sector
    group = {}
    for t, sc in zip(tickers, sec_cols, strict=False):
        group.setdefault(sc, []).append(t)
    sec_df = (
        pd.DataFrame({sc: w[cols].sum(axis=1) for sc, cols in group.items()})
        .reindex(index=w.index)
        .fillna(0.0)
    )

    # rolling mean exposures
    sec_roll = sec_df.rolling(window, min_periods=minp).mean()
    return sec_roll


# ---------- factor proxies ----------
def momentum_12_1_proxy(returns: pd.Series, window: int = 252) -> pd.Series:
    """
    12-1 momentum proxy: trailing 12M cumulative return minus last 1M cumulative return.
    Simple approximation using rolling compounded returns.
    """
    r = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    roll_12m = (1.0 + r).rolling(window, min_periods=window).apply(
        np.prod, raw=True
    ) - 1.0
    roll_1m = (1.0 + r).rolling(21, min_periods=21).apply(np.prod, raw=True) - 1.0
    return roll_12m - roll_1m


def quality_inverse_downside_vol(
    returns: pd.Series,
    window: int = 63,
    min_periods: int | None = None,
    mar: float = 0.0,
    trading_days: int = 252,
) -> pd.Series:
    """Quality proxy = inverse downside deviation (higher is better)."""
    minp = window if min_periods is None else int(min_periods)
    r = pd.to_numeric(returns, errors="coerce")
    downside = r.copy()
    downside[downside > mar / trading_days] = mar / trading_days
    dd = downside.sub(mar / trading_days).clip(upper=0.0) ** 2
    ddev = dd.rolling(window, min_periods=minp).mean().pow(0.5)
    inv = 1.0 / ddev.replace(0.0, np.nan)
    return inv


# ---------- combined builder ----------
class FactorOutputs:
    df: pd.DataFrame
    last_date: pd.Timestamp | None
    meta: dict[str, float]


def build_factor_exposures(
    weights_df: pd.DataFrame,
    returns_series: pd.Series,
    mapping: dict[str, str],
    window: int = 63,
) -> FactorOutputs:
    """
    Returns FactorOutputs with columns:
      - sector_* (rolling sector exposures)
      - mom_12_1_proxy
      - quality_inv_downside_vol
    """
    sec = rolling_sector_exposures_from_weights(weights_df, mapping, window=window)
    mom = momentum_12_1_proxy(returns_series)
    qual = quality_inverse_downside_vol(returns_series, window=window)

    out = pd.DataFrame(index=sec.index.union(mom.index).union(qual.index))
    out = out.join(sec, how="left")
    out["mom_12_1_proxy"] = mom
    out["quality_inv_downside_vol"] = qual

    last_date = pd.to_datetime(out.index).max() if len(out.index) else None
    meta = {
        "window": float(window),
        "rows": float(len(out)),
    }
    fo = FactorOutputs()
    fo.df = out
    fo.last_date = last_date
    fo.meta = meta
    return fo

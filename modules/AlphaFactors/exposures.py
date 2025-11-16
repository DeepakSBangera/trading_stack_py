from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# ---------- sector mapping ----------


def load_sector_mapping(csv_path: Path | str) -> dict[str, str]:
    """
    CSV format:
        ticker,sector
        INFY.NS,IT
        HDFCBANK.NS,BANKS
    """
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"Sector mapping file not found: {p}")
    df = pd.read_csv(p)
    cols = {c.strip().lower(): c for c in df.columns}
    if "ticker" not in cols or "sector" not in cols:
        raise ValueError("Mapping must have columns: ticker, sector")
    tick = df[cols["ticker"]].astype(str).str.strip()
    sect = df[cols["sector"]].astype(str).str.strip()
    return dict(zip(tick, sect, strict=False))


def _normalize_ticker_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def rolling_sector_exposures_from_weights(
    weights: pd.DataFrame,
    mapping: dict[str, str],
    window: int = 63,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """
    Aggregate portfolio weights into sector exposures over a rolling window
    by summing weights of all tickers that map to a sector.

    Robustness:
      - If a 'date' column is present, move it to the DatetimeIndex.
      - Coerce all columns to numeric; non-numeric -> NaN (ignored in sums).
      - Unknown tickers map to 'Unknown'.
      - Return a tz-naive DatetimeIndex to avoid join errors.
    """
    if weights.empty:
        return pd.DataFrame(index=weights.index)

    W = _normalize_ticker_columns(weights)

    # If a 'date' column slipped in, promote it to index and drop the column
    if "date" in W.columns and not isinstance(W.index, pd.DatetimeIndex):
        idx = pd.to_datetime(W["date"], errors="coerce", utc=True)
        if idx.notna().any():
            W = W.drop(columns=["date"])
            W.index = idx

    # Coerce everything to numeric (non-numeric -> NaN)
    W = W.apply(pd.to_numeric, errors="coerce")

    # Build sector -> list[ticker] mapping
    sectors = [mapping.get(t, "Unknown") for t in W.columns]
    bucket: dict[str, list[str]] = {}
    for col, sec in zip(W.columns, sectors, strict=False):
        bucket.setdefault(sec, []).append(col)

    # Sum ticker weights inside each sector (NaNs ignored by sum)
    sector_df = pd.DataFrame({sec: W[cols].sum(axis=1) for sec, cols in bucket.items()}, index=W.index)

    # Smooth with rolling mean
    out = sector_df.rolling(window=window, min_periods=min_periods or 1).mean()

    # **Normalize index to tz-naive** (strip timezone so joins don't fail)
    if isinstance(out.index, pd.DatetimeIndex) and out.index.tz is not None:
        out.index = out.index.tz_localize(None)

    return out


# ---------- factor proxies ----------


def _pick_trading_days(default_days: int, **kwargs) -> int:
    """
    Backward compatibility helper:
    - accept 'trading_days_12m' (legacy) OR 'trading_days' (new)
    - if both provided, 'trading_days' wins
    """
    if "trading_days" in kwargs and kwargs["trading_days"] is not None:
        return int(kwargs["trading_days"])
    if "trading_days_12m" in kwargs and kwargs["trading_days_12m"] is not None:
        return int(kwargs["trading_days_12m"])
    return int(default_days)


def momentum_12_1_proxy(
    nav: pd.Series,
    *,
    window_months: int = 12,
    skip_months: int = 1,
    trading_days: int | None = None,
    min_periods: int | None = None,
    **kwargs,
) -> pd.Series:
    """
    12-1 momentum proxy on NAV:
      - Exclude the most recent 'skip_months'
      - Compute total return over the prior 'window_months'
    Implementation detail (daily data):
      - Convert months to days via trading_days (default 252)
      - Shift by skip_days, then pct_change over lookback_days
    Accepts legacy kw 'trading_days_12m' for compatibility.
    """
    td = _pick_trading_days(trading_days or 252, **kwargs)
    lookback_days = max(1, round(window_months * td / 12))
    skip_days = max(0, round(skip_months * td / 12))

    s = pd.to_numeric(nav, errors="coerce")
    shifted = s.shift(skip_days)
    # Total return over lookback window (e.g., P_{t-skip} / P_{t-skip-lookback} - 1)
    # Use fill_method=None to avoid pandas FutureWarning.
    mom = shifted.pct_change(periods=lookback_days, fill_method=None)
    if min_periods is not None:
        mom = mom.where(mom.notna().rolling(min_periods).sum() == min_periods)
    return mom


def quality_inverse_downside_vol(
    returns: pd.Series,
    *,
    window: int = 63,
    min_periods: int | None = None,
    mar: float = 0.0,
    trading_days: int = 252,
    **_kwargs,
) -> pd.Series:
    """
    Quality proxy = inverse downside deviation over a rolling window.
      - downside deviation uses only returns below MAR (minimum acceptable return)
      - higher is better (inverse of volatility of bad returns)
    """
    r = pd.to_numeric(returns, errors="coerce")
    # mask: only downside
    downside = (r - (mar / trading_days)).clip(upper=0.0)
    # rolling sqrt(mean(square(downside)))
    dd = downside.pow(2).rolling(window=window, min_periods=min_periods or 1).mean().pow(0.5)
    q = 1.0 / dd.replace(0.0, np.nan)
    return q


# ---------- packaging helpers ----------


@dataclass
class FactorOutputs:
    df: pd.DataFrame
    last_date: pd.Timestamp | None
    meta: dict[str, float]

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

DATE_CANDIDATES = ("date", "Date", "timestamp", "dt")
NAV_CANDIDATES = ("nav_net", "nav", "NAV", "equity", "cumret", "cum_ret")
RET_CANDIDATES = ("ret_net", "ret", "returns", "daily_return", "r")


def _find_first(df: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    for n in names:
        if n in df.columns:
            return n
    return None


def _ensure_utc(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s, errors="coerce")
    if s.dt.tz is None:
        # make tz-aware without shifting; we don't care about wall-clock here
        s = s.dt.tz_localize("UTC")
    else:
        s = s.dt.tz_convert("UTC")
    return s


def _nav_from_returns(rets: pd.Series, start_nav: float = 1.0) -> pd.Series:
    rets = (
        pd.to_numeric(rets, errors="coerce").fillna(0.0).replace([np.inf, -np.inf], 0.0)
    )
    nav = (1.0 + rets).cumprod() * float(start_nav)
    # Safety for flat series
    nav = nav.replace([np.inf, -np.inf], np.nan).ffill().fillna(start_nav)
    return nav.astype(float)


def load_equity(path: str | pathlib.Path) -> pd.DataFrame:
    """
    Load a portfolio equity Parquet into a normalized frame:

    Columns:
      - date  (tz-aware UTC)
      - _nav  (float, clean)
      - _ret  (simple daily return, float)

    Logic:
      1) Prefer an existing nav column (nav_net > nav > …)
      2) Else compute NAV from a returns column if available
      3) Require a date-like column; raise if none
    """
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"missing file: {p}")

    df = pd.read_parquet(p)

    # --- date ---
    date_col = _find_first(df, DATE_CANDIDATES)
    if date_col is None:
        # Sometimes the index is the date
        if isinstance(df.index, (pd.DatetimeIndex, pd.PeriodIndex)):
            d = df.index.to_series()
        else:
            raise ValueError("equity file missing 'date'")
    else:
        d = df[date_col]

    d = _ensure_utc(pd.Series(d).reset_index(drop=True))
    # Filter out bad rows
    mask = d.notna()
    d = d[mask]
    df = df.loc[mask.values].reset_index(drop=True)

    # --- nav/ret ---
    nav_col = _find_first(df, NAV_CANDIDATES)
    ret_col = _find_first(df, RET_CANDIDATES)

    if nav_col is not None:
        nav = (
            pd.to_numeric(df[nav_col], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .ffill()
            .fillna(1.0)
        )
        nav = nav.astype(float)
        ret = nav.pct_change().fillna(0.0)
    elif ret_col is not None:
        ret = (
            pd.to_numeric(df[ret_col], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        nav = _nav_from_returns(ret)
    else:
        # Fallback: some earlier files had 'ret_gross' / 'ret_net'
        if "ret_gross" in df.columns:
            ret = (
                pd.to_numeric(df["ret_gross"], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
            )
            nav = _nav_from_returns(ret)
        elif "ret_net" in df.columns:
            ret = (
                pd.to_numeric(df["ret_net"], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
            )
            nav = _nav_from_returns(ret)
        else:
            raise ValueError("no NAV/return columns found")

    out = pd.DataFrame(
        {
            "date": d.values,
            "_nav": nav.values.astype(float),
            "_ret": ret.values.astype(float),
        }
    )
    # drop duplicate dates; keep last
    out = (
        out.sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )
    return out

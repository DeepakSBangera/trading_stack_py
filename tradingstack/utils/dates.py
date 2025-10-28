from __future__ import annotations

from typing import Iterable

import pandas as pd


def to_naive_date_index(x: pd.Series | pd.Index | Iterable) -> pd.DatetimeIndex:
    """
    Convert anything date-like into a tz-naive daily DatetimeIndex (midnight).
    Works for Series and Index without using `.dt` on an Index.
    """
    s = pd.to_datetime(x, errors="coerce", utc=True)
    # Remove tz if present
    for fn in ("tz_convert", "tz_localize"):
        try:
            s = getattr(s, fn)(None)
        except Exception:
            pass
    try:
        s = s.floor("D")
    except Exception:
        s = pd.to_datetime(pd.Series(s), errors="coerce").dt.floor("D")
    return pd.DatetimeIndex(s)


def normalize_date_series(s: pd.Series) -> pd.Series:
    """Return tz-naive daily Series (values preserved)."""
    idx = to_naive_date_index(s)
    out = pd.Series(s.values, index=idx)
    return out


def coerce_date_index(df: pd.DataFrame, date_col: str | None = "date") -> pd.DataFrame:
    """
    Ensure a tz-naive daily DatetimeIndex. If `date_col` exists, use it;
    otherwise attempt to parse the index.
    """
    out = df.copy()
    if date_col and date_col in out.columns:
        out.index = to_naive_date_index(out[date_col])
        out = out.drop(columns=[date_col])
    else:
        out.index = to_naive_date_index(out.index)
    return out.sort_index()

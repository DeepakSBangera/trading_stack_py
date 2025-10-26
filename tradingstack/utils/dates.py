from __future__ import annotations

import pandas as pd


def normalize_date_index(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    Ensures a timezone-naive, ascending-sorted DatetimeIndex named 'date'.
    Accepts either a 'date' column or an existing index.
    - Casts to pandas datetime, strips tz (if any), sorts, de-duplicates by date.
    """
    if date_col in df.columns:
        dt = pd.to_datetime(df[date_col], utc=False, errors="coerce")
        if getattr(dt.dt, "tz", None) is not None:
            dt = dt.dt.tz_localize(None)
        df = df.assign(**{date_col: dt}).dropna(subset=[date_col])
        df = df.sort_values(date_col).drop_duplicates(subset=[date_col])
        df = df.set_index(date_col)
    else:
        idx = pd.to_datetime(df.index, utc=False, errors="coerce")
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
        df.index = idx
        df = df.dropna(axis=0, how="any").sort_index()

    df.index.name = "date"
    return df

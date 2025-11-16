from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def _ensure_dt_index(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        cols = {c.lower(): c for c in df.columns}
        if date_col.lower() in cols:
            dcol = cols[date_col.lower()]
            idx = pd.to_datetime(df[dcol], utc=True, errors="coerce")
            df = df.drop(columns=[dcol])
            df.index = idx
        else:
            raise TypeError("Need a datetime column or DatetimeIndex")
    df.index = pd.to_datetime(df.index, utc=True)
    return df.sort_index()


def pivot_weights(w: pd.DataFrame) -> pd.DataFrame:
    """Return weights table indexed by date, columns=tickers, numeric, NaN->0."""
    w = _ensure_dt_index(w)
    out = w.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return out.loc[:, ~out.columns.duplicated(keep="last")]


def build_returns_from_prices(px: pd.DataFrame) -> pd.DataFrame:
    """Simple returns per ticker from prices; forward-fill gaps minimally."""
    px = _ensure_dt_index(px)
    px = px.apply(pd.to_numeric, errors="coerce")
    px = px.replace([np.inf, -np.inf], np.nan).ffill(limit=3)
    rets = px.pct_change()
    return rets.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def align_weights_and_returns(w: pd.DataFrame, r: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align on common dates & tickers."""
    w = pivot_weights(w)
    r = build_returns_from_prices(r) if "open" in r.columns or "close" in r.columns else _ensure_dt_index(r)
    common_cols = sorted(set(w.columns) & set(r.columns))
    common_idx = w.index.intersection(r.index)
    return w.loc[common_idx, common_cols], r.loc[common_idx, common_cols]


def contribution_by_ticker(w: pd.DataFrame, r: pd.DataFrame) -> pd.DataFrame:
    """Daily contribution by ticker = weight * return."""
    w_al, r_al = align_weights_and_returns(w, r)
    return w_al * r_al


def group_contribution(contrib: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    """Group ticker contributions to groups via mapping dict {ticker->group}."""
    contrib = contrib.copy()
    groups = {}
    for t in contrib.columns:
        g = mapping.get(t, "Unknown")
        groups.setdefault(g, []).append(t)
    out = pd.DataFrame(index=contrib.index)
    for g, cols in groups.items():
        out[g] = contrib[cols].sum(axis=1)
    return out

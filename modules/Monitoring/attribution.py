# tradingstack/metrics/attribution.py
from __future__ import annotations

import pathlib
from collections.abc import Iterable

import numpy as np
import pandas as pd

# ---------- Date/Index utilities ----------


def _as_dt_index(obj) -> pd.DatetimeIndex:
    """
    Coerce obj (Series/Index/iterable) into a tz-naive, ascending DatetimeIndex.
    Strategy:
      - parse with utc=True (forces tz-aware),
      - drop tz via tz_convert(None),
      - sort and return.
    This avoids calling tz_convert on a tz-naive index.
    """
    # Parse anything into a tz-aware datetime index
    dt = pd.to_datetime(obj, errors="coerce", utc=True)

    # Normalize to a DatetimeIndex
    if isinstance(dt, pd.Series):
        dt = pd.DatetimeIndex(dt)
    elif not isinstance(dt, pd.DatetimeIndex):
        dt = pd.DatetimeIndex(dt)

    # Drop timezone to tz-naive
    # At this point dt is tz-aware because we used utc=True, so this is always safe.
    dt = dt.tz_convert(None)

    # Sort and return
    return dt.sort_values()


def _ensure_dt_index_on_frame(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    Ensure df has a tz-naive DatetimeIndex. If 'date' column exists, use it.
    Otherwise, parse df.index. Drops NaT rows, sorts ascending.
    """
    if date_col in df.columns:
        idx = _as_dt_index(df[date_col])
    else:
        idx = _as_dt_index(df.index)

    mask = ~pd.isna(idx)
    out = df.loc[mask].copy()
    out.index = idx[mask]
    out.index.name = "date"
    return out.sort_index()


# ---------- Weights ----------


def pivot_weights(weights_long: pd.DataFrame) -> pd.DataFrame:
    """
    Input (long): columns = ['date','ticker','weight'] (case-insensitive ok)
    Output (wide): Date index (tz-naive), columns are tickers, values are weights.
    """
    cols_lower = {c.lower(): c for c in weights_long.columns}
    required = {"date", "ticker", "weight"}
    missing = required - set(cols_lower.keys())
    if missing:
        raise ValueError(f"weights missing columns: {sorted(missing)}")

    df = weights_long.rename(
        columns={
            cols_lower["date"]: "date",
            cols_lower["ticker"]: "ticker",
            cols_lower["weight"]: "weight",
        }
    ).copy()

    df = _ensure_dt_index_on_frame(df, "date")
    df["ticker"] = df["ticker"].astype(str)

    wide = df.pivot_table(index=df.index, columns="ticker", values="weight", aggfunc="last").sort_index()
    # Forward-fill weights across days; missing = 0
    wide = wide.ffill().fillna(0.0)
    wide.columns.name = None
    wide.index.name = "date"
    return wide


# ---------- Price -> Returns ----------


def _load_one_price_series(prq: pathlib.Path) -> pd.Series:
    """
    Load a single ticker parquet file and return a tz-naive daily return Series.
    Accepts either:
      - 'ret' column (already returns), or
      - 'close' column (compute pct_change).
    Tolerates 'date' column or datetime index in file.
    """
    df = pd.read_parquet(prq)
    df = _ensure_dt_index_on_frame(df)  # set/clean DatetimeIndex

    # Prefer an explicit 'ret' column (case-insensitive)
    ret_cols = [c for c in df.columns if c.lower() == "ret"]
    if ret_cols:
        s = pd.to_numeric(df[ret_cols[0]], errors="coerce")
    else:
        # Try to compute from close-like columns
        close_candidates = [c for c in df.columns if c.lower() in ("close", "adj_close", "adjclose", "price")]
        if close_candidates:
            px = pd.to_numeric(df[close_candidates[0]], errors="coerce").replace([np.inf, -np.inf], np.nan).ffill()
            s = px.pct_change().fillna(0.0)
        else:
            # No usable column; return zeros
            s = pd.Series(0.0, index=df.index)

    s = s.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    s.index = _as_dt_index(s.index)
    s = s[~s.index.isna()].sort_index()
    s.index.name = "date"
    return s


def build_returns_from_prices(root: pathlib.Path, tickers: Iterable[str]) -> pd.DataFrame:
    """
    Given a directory with parquet files per ticker, return a wide
    DataFrame of daily returns indexed by date (tz-naive), columns=tickers.
    Looks for filenames like <TICKER>.parquet and a few simple variants.
    """
    root = pathlib.Path(root)
    series = {}
    for t in tickers:
        candidates = [
            root / f"{t}.parquet",
            root / f"{t.replace('.', '_')}.parquet",
            root / f"{t.replace('.', '')}.parquet",
            root / f"{t.upper()}.parquet",
            root / f"{t.lower()}.parquet",
        ]
        prq = next((p for p in candidates if p.exists()), None)
        if prq is None:
            series[t] = pd.Series(dtype=float)
            continue
        s = _load_one_price_series(prq)
        series[t] = s

    if not series:
        return pd.DataFrame()

    r_wide = pd.DataFrame(series).sort_index()
    r_wide.index = _as_dt_index(r_wide.index)
    r_wide = r_wide.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    r_wide.index.name = "date"
    return r_wide


# ---------- Alignment ----------


def align_weights_and_returns(
    wide_weights: pd.DataFrame, wide_returns: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align weights and returns to a common tz-naive date index and ticker set.
    We forward-fill weights, leave returns as-is (missing = 0).
    """
    W = wide_weights.copy()
    R = wide_returns.copy()

    W.index = _as_dt_index(W.index)
    R.index = _as_dt_index(R.index)
    W = W.sort_index()
    R = R.sort_index()

    cols = sorted(set(W.columns).intersection(set(R.columns)))
    if not cols:
        idx = W.index.intersection(R.index)
        idx = pd.DatetimeIndex(idx)
        return (
            pd.DataFrame(index=idx),
            pd.DataFrame(index=idx),
        )

    W = W[cols]
    R = R[cols]

    idx = W.index.intersection(R.index).unique().sort_values()
    W = W.reindex(idx).ffill().fillna(0.0)
    R = R.reindex(idx).fillna(0.0)

    W.index.name = "date"
    R.index.name = "date"
    return W, R


# ---------- Contribution / Grouping ----------


def contribution_by_ticker(W: pd.DataFrame, R: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Per-ticker contributions = W.shift(1) * R
    Portfolio return = row-wise sum of contributions.
    """
    if W.empty or R.empty:
        idx = W.index if not W.empty else R.index
        idx = pd.DatetimeIndex(idx) if len(idx) else pd.DatetimeIndex([])
        return pd.DataFrame(index=idx), pd.Series(dtype=float, index=idx)

    Wlag = W.shift(1).fillna(0.0)
    contrib = (Wlag * R).fillna(0.0)
    port_ret = contrib.sum(axis=1)
    contrib.index.name = "date"
    port_ret.index.name = "date"
    return contrib, port_ret


def group_contribution(
    contrib_by_ticker: pd.DataFrame,
    mapping_csv: pathlib.Path,
    group_col: str = "sector",
) -> pd.DataFrame:
    """
    Sum contributions by a group mapping CSV with columns: ticker,<group_col>.
    Returns wide frame with columns=groups.
    """
    try:
        mp = pd.read_csv(mapping_csv)
    except Exception:
        return pd.DataFrame(index=contrib_by_ticker.index)

    # Normalize columns to lower
    mp = mp.rename(columns={c: c.lower() for c in mp.columns})
    need = {"ticker", group_col}
    if not need.issubset(set(mp.columns)):
        return pd.DataFrame(index=contrib_by_ticker.index)

    mp = mp[["ticker", group_col]].copy()
    mp["ticker"] = mp["ticker"].astype(str)

    cols = [c for c in contrib_by_ticker.columns if c in set(mp["ticker"])]
    if not cols:
        return pd.DataFrame(index=contrib_by_ticker.index)

    gmap = dict(zip(mp["ticker"], mp[group_col]))
    tmp = contrib_by_ticker[cols].copy()
    tmp.columns = pd.MultiIndex.from_tuples([(gmap.get(c, "UNK"), c) for c in tmp.columns], names=[group_col, "ticker"])
    grp = tmp.groupby(level=0, axis=1).sum()
    grp.index.name = "date"
    return grp

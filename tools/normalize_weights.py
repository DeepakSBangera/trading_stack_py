# tools/normalize_weights.py
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pandas as pd

CAND_DATE = ["date", "dt", "timestamp", "time"]
CAND_TICKER = ["ticker", "symbol", "sid", "asset", "code", "ric"]
CAND_WEIGHT = ["weight", "w", "target", "alloc", "allocation", "weight_pct", "wgt"]


def _pick(colnames, candidates):
    cols_lc = {str(c).lower(): c for c in colnames}
    for k in candidates:
        if k in cols_lc:
            return cols_lc[k]
    return None


def _try_clean_dates(s: pd.Series) -> pd.Series | None:
    """
    Parse to datetime; return tz-naive UTC values if possible, else None.
    (Use .dt.tz_convert because Series.tz_convert works on the index.)
    """
    try:
        dt = pd.to_datetime(s, errors="coerce", utc=True)
    except Exception:
        return None
    if dt.isna().all():
        return None
    return dt.dt.tz_convert("UTC").dt.tz_localize(None)


def _renorm_per_day(df):
    def _renorm(g):
        s = g["weight"].abs().sum()
        if s > 0:
            g["weight"] = g["weight"] / s
        return g

    return df.groupby(df["date"], group_keys=False).apply(_renorm)


def _load_equity_dates(
    equity_path: pathlib.Path, nrows_expected: int | None = None
) -> pd.Series:
    eq = pd.read_parquet(equity_path)
    if "date" not in eq.columns:
        raise ValueError(f"equity file missing 'date': {equity_path}")
    d = pd.to_datetime(eq["date"], errors="coerce", utc=True)
    d = d.dt.tz_convert("UTC").dt.tz_localize(None)
    if d.isna().any():
        raise ValueError(f"equity 'date' has NaT values: {equity_path}")
    if nrows_expected is not None and len(d) != nrows_expected:
        raise ValueError(
            f"row count mismatch: weights={nrows_expected} vs equity dates={len(d)} ({equity_path})"
        )
    return d


def normalize(df: pd.DataFrame, equity_dates: pd.Series | None) -> pd.DataFrame:
    # Detect LONG first
    dcol = _pick(df.columns, CAND_DATE)
    tcol = _pick(df.columns, CAND_TICKER)
    wcol = _pick(df.columns, CAND_WEIGHT)

    if dcol and tcol and wcol:
        # LONG → standardize
        date_series = _try_clean_dates(df[dcol])
        if date_series is None and equity_dates is not None:
            if len(df) != len(equity_dates):
                raise ValueError(
                    f"row mismatch: long-weights={len(df)} vs equity dates={len(equity_dates)}"
                )
            date_series = equity_dates.reset_index(drop=True)
        if date_series is None:
            raise TypeError(
                "Could not parse dates in long weights; provide --equity to borrow dates."
            )
        out = pd.DataFrame(
            {
                "date": date_series,
                "ticker": df[tcol].astype(str).str.strip(),
                "weight": pd.to_numeric(df[wcol], errors="coerce"),
            }
        )
    else:
        # WIDE → index = date (if present), columns = tickers
        wide = df.copy()
        if wide.index.name is None:
            wide.index.name = "date"
        wide_reset = wide.reset_index()

        date_series = _try_clean_dates(wide_reset["date"])
        if date_series is None and equity_dates is not None:
            if len(wide_reset) != len(equity_dates):
                raise ValueError(
                    f"row mismatch: wide-weights={len(wide_reset)} vs equity dates={len(equity_dates)}"
                )
            date_series = equity_dates.reset_index(drop=True)
        if date_series is None:
            raise TypeError(
                "Could not infer dates from weights (index isn’t datetime). "
                "Re-run with --equity <portfolio_v2.parquet> to borrow dates."
            )

        wide_reset["date"] = date_series
        out = wide_reset.melt(id_vars=["date"], var_name="ticker", value_name="weight")
        out["ticker"] = out["ticker"].astype(str).str.strip()
        out["weight"] = pd.to_numeric(out["weight"], errors="coerce")

    # Clean + bounds
    out = out.dropna(subset=["date", "ticker", "weight"])
    out["weight"] = out["weight"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["weight"] = out["weight"].clip(-1.0, 1.0)

    out = out.sort_values(["date", "ticker"])

    # L1 renorm per day
    out = _renorm_per_day(out)

    return out[["date", "ticker", "weight"]]


def main():
    ap = argparse.ArgumentParser(
        description="Normalize weights (wide or long) to date,ticker,weight; borrow dates from equity if needed."
    )
    ap.add_argument(
        "--in", dest="inp", required=True, help="input weights parquet (wide or long)"
    )
    ap.add_argument(
        "--out", dest="outp", required=True, help="output parquet (date,ticker,weight)"
    )
    ap.add_argument(
        "--equity",
        dest="equity",
        required=False,
        help="portfolio_v2 parquet to borrow dates from (if weights lack dates)",
    )
    args = ap.parse_args()

    src = pathlib.Path(args.inp)
    if not src.exists():
        raise FileNotFoundError(src)

    equity_dates = None
    if args.equity:
        equity_dates = _load_equity_dates(pathlib.Path(args.equity))

    df = pd.read_parquet(src)
    out = normalize(df, equity_dates)
    pathlib.Path(args.outp).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.outp, index=False)
    print(f"OK -> {args.outp} | rows={len(out)} | cols={list(out.columns)}")


if __name__ == "__main__":
    main()

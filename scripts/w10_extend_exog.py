# scripts/w10_extend_exog.py
from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

DATE_CANDS: tuple[str, ...] = ("date", "timestamp", "dt")


def pick(cols: Iterable[str], cands: Iterable[str]) -> str | None:
    lower = {str(c).lower(): c for c in cols}
    for w in cands:
        if w in lower:
            return lower[w]
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Extend/align exog over the price date range.")
    ap.add_argument("--data-glob", default="data/csv/*.csv")
    ap.add_argument("--exog-in", default="data/factors/nifty_macro.csv")
    ap.add_argument("--exog-out", default="data/factors/nifty_macro_extended.csv")
    ap.add_argument("--cols", default="mkt_ret,carry")
    args = ap.parse_args()

    # Price date bounds
    mins: list[pd.Timestamp] = []
    maxs: list[pd.Timestamp] = []

    for p in Path().glob(args.data_glob):
        try:
            df = pd.read_csv(p)
            dcol = pick(df.columns, DATE_CANDS)
            if not dcol:
                continue
            df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
            df = df.dropna(subset=[dcol]).sort_values(dcol)
            if not df.empty:
                mins.append(df[dcol].min())
                maxs.append(df[dcol].max())
        except Exception:
            continue

    if not mins:
        raise SystemExit("No dated price files found")

    lo, hi = min(mins), max(maxs)

    # Load exog
    ex = pd.read_csv(args.exog_in)
    dcol = pick(ex.columns, DATE_CANDS) or "date"
    ex[dcol] = pd.to_datetime(ex[dcol], errors="coerce")
    ex = ex.dropna(subset=[dcol]).sort_values(dcol).set_index(dcol)

    cols = [c.strip() for c in args.cols.split(",") if c.strip()]
    for c in cols:
        if c not in ex.columns:
            ex[c] = 0.0  # create neutral column if missing

    # Business-day reindex and gentle fill
    full_idx = pd.bdate_range(lo, hi)
    ex = ex.reindex(full_idx).ffill().bfill(limit=5)

    ex_out = ex[cols].reset_index().rename(columns={"index": "date"})
    ex_out.to_csv(args.exog_out, index=False)
    print(
        f"Wrote {args.exog_out} spanning {full_idx.min().date()}.."
        f"{full_idx.max().date()} ({len(full_idx)} rows)"
    )


if __name__ == "__main__":
    main()

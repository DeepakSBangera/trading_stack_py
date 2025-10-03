# scripts/w10_make_exog_stub.py
from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

DATE_CANDS: tuple[str, ...] = ("date", "timestamp", "dt")


def pick_date(cols: Iterable[str]) -> str | None:
    lower = {str(c).lower(): c for c in cols}
    for k in DATE_CANDS:
        if k in lower:
            return lower[k]
    return None


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Create a stub exog CSV aligned to first price file dates."
    )
    ap.add_argument("--data-glob", default="data/csv/*.csv")
    ap.add_argument("--out", default="data/factors/nifty_macro.csv")
    ap.add_argument("--cols", default="mkt_ret,carry")
    args = ap.parse_args()

    paths = list(Path().glob(args.data_glob))
    if not paths:
        print(f"No files matched {args.data_glob}", file=sys.stderr)
        raise SystemExit(1)

    df = pd.read_csv(paths[0])
    dcol = pick_date(df.columns) or "date"
    if dcol not in df.columns:
        # Build a sequential business-day index if no date column exists
        n = len(df)
        dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
        out = pd.DataFrame({"date": dates})
    else:
        df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
        df = df.dropna(subset=[dcol]).sort_values(dcol)
        dates = pd.bdate_range(df[dcol].min(), df[dcol].max())
        out = pd.DataFrame({"date": dates})

    for c in [c.strip() for c in args.cols.split(",") if c.strip()]:
        out[c] = 0.0

    out.to_csv(args.out, index=False)
    print(f"Wrote stub exog: {args.out} ({len(out)} rows)")


if __name__ == "__main__":
    main()

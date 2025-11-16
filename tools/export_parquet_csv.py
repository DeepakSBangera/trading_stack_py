from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser(description="Export a parquet file to CSV with optional row limit.")
    ap.add_argument("parquet", nargs="?", default="reports/factor_exposures.parquet")
    ap.add_argument("--out", default=None, help="Output CSV path (default: alongside parquet)")
    ap.add_argument(
        "--limit",
        type=int,
        default=5000,
        help="Max rows to export (0 or negative = all)",
    )
    ap.add_argument("--index", action="store_true", help="Keep DataFrame index in CSV")
    args = ap.parse_args()

    p = Path(args.parquet)
    if not p.exists():
        raise SystemExit(f"Input parquet not found: {p}")

    df = pd.read_parquet(p)

    if args.limit and args.limit > 0 and len(df) > args.limit:
        df = df.head(args.limit)

    out = Path(args.out) if args.out else p.with_suffix(".csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=args.index)
    print(f"[OK] Wrote CSV: {out.resolve()}")


if __name__ == "__main__":
    main()

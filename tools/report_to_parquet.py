import argparse
import glob
import os

import pandas as pd

SKIP_PREFIXES = ("wk",)  # governance summaries often start with "wk"; we skip them by default


def convert_one(path: str, delete_csv: bool) -> str:
    df = pd.read_csv(path)
    # coerce helpful types
    for c in df.columns:
        cl = c.lower()
        if cl in ("date", "asof", "timestamp", "time"):
            df[c] = pd.to_datetime(df[c], errors="coerce")
    out = os.path.splitext(path)[0] + ".parquet"
    df.to_parquet(out, index=False, compression="snappy")
    if delete_csv:
        try:
            os.remove(path)
        except Exception as e:
            print(f"warn: could not delete {path}: {e}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default="reports")
    ap.add_argument(
        "--include-governance",
        action="store_true",
        help="Also convert CSVs like *_tearsheet.csv or wk*.csv",
    )
    ap.add_argument("--delete-csv", action="store_true")
    args = ap.parse_args()

    pats = ["portfolioV2_*.csv"]  # main payloads
    if args.include_governance:
        pats += ["*_tearsheet.csv", "wk*.csv"]

    to_convert = []
    for pat in pats:
        to_convert += glob.glob(os.path.join(args.reports, pat))
    # De-dupe; optionally skip governance if flag not set
    to_convert = sorted(set(to_convert))
    if not args.include_governance:
        to_convert = [p for p in to_convert if os.path.basename(p).startswith("portfolioV2_")]
    if not to_convert:
        print("No matching CSVs found.")
        return

    print(f"Converting {len(to_convert)} CSVs → Parquet (delete={args.delete_csv})...")
    for f in to_convert:
        out = convert_one(f, args.delete_csv)
        print(f"✓ {os.path.basename(f)} → {os.path.basename(out)}")
    print("Done.")


if __name__ == "__main__":
    main()

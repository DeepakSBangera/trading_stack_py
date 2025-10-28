from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="reports/weights_v2_norm.parquet")
    ap.add_argument("--out", default="config/sector_mapping.csv")
    args = ap.parse_args()

    w = Path(args.weights)
    out = Path(args.out)
    if not w.exists():
        raise SystemExit(f"weights parquet not found: {w}")

    df = pd.read_parquet(w)
    if not {"ticker"}.issubset(df.columns):
        raise SystemExit("weights parquet must have a 'ticker' column")

    tickers = sorted(set(df["ticker"].dropna().astype(str)))
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ticker": tickers, "sector": ["Unknown"] * len(tickers)}).to_csv(
        out, index=False
    )
    print(f"[OK] Wrote template: {out}")


if __name__ == "__main__":
    main()

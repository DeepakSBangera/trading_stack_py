# ruff: noqa: E402
from __future__ import annotations

import pathlib

# --- make repo root importable ---
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ---------------------------------

import argparse
import os

import numpy as np
import pandas as pd

from tradingstack.io import load_equity
from tradingstack.metrics import sharpe_annual


def metrics_from_equity(eq: pd.DataFrame, ann_factor: float = 252.0) -> dict:
    df = eq.sort_values("date").reset_index(drop=True).copy()
    nav = (
        pd.to_numeric(df["_nav"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
        .fillna(1.0)
    )
    ret = nav.pct_change().fillna(0.0)

    days = int(len(nav))
    if days <= 0:
        return dict(days=0, cagr=0.0, sharpe=0.0, max_dd=0.0, total_ret=0.0)

    nav0, navN = float(nav.iloc[0]), float(nav.iloc[-1])
    total_ret = (navN / nav0) - 1.0
    years = max(days / ann_factor, 1e-9)
    cagr = (navN / nav0) ** (1.0 / years) - 1.0
    sr = sharpe_annual(ret, ann_factor=ann_factor)

    run_max = nav.cummax()
    dd = (nav / run_max) - 1.0
    max_dd = float(dd.min())

    return dict(
        days=days,
        cagr=float(cagr),
        sharpe=float(sr),
        max_dd=max_dd,
        total_ret=float(total_ret),
    )


def main():
    ap = argparse.ArgumentParser(description="fallback comparator")
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ann-factor", type=float, default=252.0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    a = load_equity(args.a)
    b = load_equity(args.b)

    ma = metrics_from_equity(a, ann_factor=args.ann_factor)
    mb = metrics_from_equity(b, ann_factor=args.ann_factor)

    tbl = pd.DataFrame.from_dict({"A_older": ma, "B_newer": mb}, orient="index")
    tbl = tbl[["days", "cagr", "sharpe", "max_dd", "total_ret"]]

    tbl.to_parquet(os.path.join(args.out, "compare_runs.parquet"), index=True)
    tbl.to_csv(os.path.join(args.out, "compare_runs.csv"), index=True)

    print("Compare written to", args.out, "(compare_runs.*)")
    print("Summary (ASCII):")
    print(tbl.to_string(float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()

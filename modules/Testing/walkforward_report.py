from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd
from trading_stack_py.cv.walkforward import WalkForwardCV
from trading_stack_py.metrics.dsr import (
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
)

from trading_stack_py.metrics.mtl import minimum_track_record_length
from trading_stack_py.utils.returns import to_excess_returns


def _infer_returns(df: pd.DataFrame, price_col: str | None) -> tuple[np.ndarray, str]:
    # Priority 1: an explicit 'returns' column
    for cand in ["returns", "ret", "r", "daily_return", "log_return"]:
        if cand in df.columns:
            r = df[cand].astype(float).to_numpy()
            return r, cand
    # Priority 2: compute from price column if present
    if price_col and price_col in df.columns:
        p = pd.to_numeric(df[price_col], errors="coerce").ffill()
        r = p.pct_change().to_numpy()
        return r, f"{price_col}_pct_change"
    # Priority 3: try common price column names
    for cand in ["close", "adj_close", "price", "Close", "Adj Close"]:
        if cand in df.columns:
            p = pd.to_numeric(df[cand], errors="coerce").ffill()
            r = p.pct_change().to_numpy()
            return r, f"{cand}_pct_change"
    raise ValueError("Could not infer returns. Provide a 'returns' column or a valid --price-col.")


def _segment_sr(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size <= 1:
        return 0.0
    sd = x.std(ddof=1)
    return 0.0 if sd == 0 else float(x.mean() / sd)


def main():
    ap = argparse.ArgumentParser(description="Walk-forward CV report with Embargo + DSR + MTL")
    ap.add_argument("--csv", required=True, help="Input CSV with 'returns' or prices.")
    ap.add_argument("--date-col", default=None, help="Name of date column (optional).")
    ap.add_argument(
        "--price-col",
        default=None,
        help="If no returns column, compute from this price column.",
    )
    ap.add_argument(
        "--rf",
        type=float,
        default=0.0,
        help="Annual risk-free rate (e.g., 0.05 for 5%).",
    )
    ap.add_argument(
        "--freq",
        default="D",
        choices=["D", "B", "W", "M", "Q"],
        help="Return frequency for RF deflation.",
    )
    ap.add_argument("--train", type=int, default=378, help="Train size (observations).")
    ap.add_argument("--test", type=int, default=63, help="Test size (observations).")
    ap.add_argument("--step", type=int, default=None, help="Step size (default: == test).")
    ap.add_argument(
        "--expanding",
        action="store_true",
        help="Use expanding window (default: rolling).",
    )
    ap.add_argument(
        "--min-train",
        type=int,
        default=None,
        help="Minimum allowed training size (default: == train).",
    )
    ap.add_argument(
        "--embargo",
        type=int,
        default=5,
        help="Embargo between train end and test start.",
    )
    ap.add_argument(
        "--pstar",
        type=float,
        default=0.95,
        help="Confidence level for MTL (e.g., 0.95).",
    )
    ap.add_argument("--tag", default="default", help="Run tag to name the report folder.")
    ap.add_argument("--outdir", default="reports/W5", help="Root output dir.")
    args = ap.parse_args()

    # Load data
    df = pd.read_csv(args.csv)
    if args.date_col and args.date_col in df.columns:
        # Keep the date column for traceability; not mandatory
        df = df.sort_values(args.date_col)

    raw_r, source = _infer_returns(df, args.price_col)
    # Deflate by RF and clean NaNs
    r = to_excess_returns(raw_r, rf=args.rf, freq=args.freq)
    r = r[~np.isnan(r)]
    n = int(r.size)
    if n < (args.train + args.test):
        raise ValueError(f"Not enough data: have {n}, need at least train+test={args.train + args.test}.")

    cv = WalkForwardCV(
        train_size=args.train,
        test_size=args.test,
        step_size=args.step,
        expanding=args.expanding,
        min_train_size=args.min_train,
        embargo=args.embargo,
    )

    rows = []
    seg_srs = []
    seg = 0
    for tr, te in cv.split(n):
        seg += 1
        r_te = r[te]
        sr = _segment_sr(r_te)
        psr = probabilistic_sharpe_ratio(sr_hat=sr, sr_threshold=0.0, n=len(r_te), skewness=0.0, kurt=3.0)
        rows.append(
            {
                "segment": seg,
                "train_start": int(tr[0]),
                "train_end": int(tr[-1]),
                "test_start": int(te[0]),
                "test_end": int(te[-1]),
                "test_len": int(len(r_te)),
                "test_sr": float(sr),
                "test_psr_vs_0": float(psr),
            }
        )
        seg_srs.append(sr)

    sr_arr = np.array(seg_srs, dtype=float)
    dsr_val = deflated_sharpe_ratio(sr_arr, num_trials=len(sr_arr))
    mean_sr = float(np.nanmean(sr_arr)) if len(sr_arr) else float("nan")
    mtl_90 = minimum_track_record_length(mean_sr, p_star=0.90)
    mtl_95 = minimum_track_record_length(mean_sr, p_star=args.pstar)

    ts = time.strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(args.outdir, f"{args.tag}_{ts}")
    os.makedirs(outdir, exist_ok=True)

    # Save per-segment metrics
    df_out = pd.DataFrame(rows)
    csv_path = os.path.join(outdir, "segment_metrics.csv")
    df_out.to_csv(csv_path, index=False)

    # Write Markdown summary
    md_path = os.path.join(outdir, "README.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# W5 Walk-Forward Report\n\n")
        f.write(f"- Data source: **{os.path.basename(args.csv)}** (returns source: **{source}**)\n")
        f.write(f"- Observations used: **{n}**\n")
        f.write(
            f"- Train/Test/Step/Expanding/Embargo: **{args.train} / {args.test} / {args.step or args.test} / {args.expanding} / {args.embargo}**\n"
        )
        f.write(f"- Segments: **{len(sr_arr)}**\n")
        f.write(f"- Mean segment SR: **{mean_sr:.3f}**\n")
        f.write(f"- DSR over segment SRs (N={len(sr_arr)}): **{dsr_val:.3f}**\n")
        f.write(f"- MTL (PSR≥90%): **{mtl_90:.1f} periods**\n")
        f.write(f"- MTL (PSR≥{args.pstar * 100:.0f}%): **{mtl_95:.1f} periods**\n\n")
        f.write("## Segments (first 10)\n\n")
        f.write(df_out.head(10).to_markdown(index=False))
        f.write("\n")

    print("Report written:")
    print(" -", csv_path)
    print(" -", md_path)


if __name__ == "__main__":
    main()

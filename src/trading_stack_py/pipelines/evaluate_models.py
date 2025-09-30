from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd


def detect_task(df: pd.DataFrame) -> str:
    if {"mse", "r2"}.issubset(df.columns):
        return "regression"
    if {"auc", "acc"}.issubset(df.columns):
        return "classification"
    # fallback: look for typical columns
    if "mse" in df.columns or "r2" in df.columns:
        return "regression"
    return "classification"


def main():
    ap = argparse.ArgumentParser(
        description="W9: Evaluate W7 models against W6 segments, write summary."
    )
    ap.add_argument(
        "--w6-dir", required=True, help="Path to W6 output (has segments.csv, segment_*/)"
    )
    ap.add_argument(
        "--w7-dir",
        required=True,
        help="Path to W7 output (has segment_metrics.csv, maybe cum_pnl_proxy.csv)",
    )
    ap.add_argument("--tag", default="W9")
    ap.add_argument("--outdir", default="reports/W9")
    args = ap.parse_args()

    w6 = Path(args.w6_dir)
    w7 = Path(args.w7_dir)
    seg6 = pd.read_csv(w6 / "segments.csv")
    met7 = pd.read_csv(w7 / "segment_metrics.csv")

    # standardize column names
    seg6 = seg6.rename(columns={c: c.lower() for c in seg6.columns})
    met7 = met7.rename(columns={c: c.lower() for c in met7.columns})

    # join on segment
    df = pd.merge(seg6, met7, on="segment", how="inner")

    # prefer test_len from either side
    if "test_len_x" in df and "test_len_y" in df:
        df["n_test"] = df["test_len_y"].fillna(df["test_len_x"])
        df = df.drop(columns=[c for c in ["test_len_x", "test_len_y"] if c in df])
    elif "test_len" in df:
        df["n_test"] = df["test_len"]
    elif "n_test" in df:
        pass
    else:
        # derive if absent
        if {"test_start", "test_end"}.issubset(df.columns):
            df["n_test"] = df["test_end"] - df["test_start"] + 1

    task = detect_task(df)

    # aggregates
    lines = []
    if task == "regression":
        mse_med = df["mse"].median(skipna=True) if "mse" in df else np.nan
        r2_med = df["r2"].median(skipna=True) if "r2" in df else np.nan
        mse_w = np.nan
        if {"mse", "n_test"}.issubset(df.columns):
            w = df["n_test"].astype(float)
            mse_w = (df["mse"] * w).sum() / w.sum()
        pnl_sum = df["pnl_proxy"].sum() if "pnl_proxy" in df else np.nan
        lines += [
            "- Task: **regression**",
            f"- Segments: **{len(df)}**",
            f"- MSE (median): **{mse_med:.6f}**",
            f"- MSE (weighted by test size): **{mse_w:.6f}**",
            f"- R² (median): **{r2_med:.6f}**",
            f"- PnL proxy (sum): **{pnl_sum:.4f}**",
        ]
    else:
        # classification
        auc_med = df["auc"].median(skipna=True) if "auc" in df else np.nan
        acc_med = df["acc"].median(skipna=True) if "acc" in df else np.nan
        prec_med = df["precision"].median(skipna=True) if "precision" in df else np.nan
        rec_med = df["recall"].median(skipna=True) if "recall" in df else np.nan
        pnl_sum = df["pnl_proxy"].sum() if "pnl_proxy" in df else np.nan
        lines += [
            "- Task: **classification**",
            f"- Segments: **{len(df)}**",
            f"- AUC (median): **{auc_med:.4f}**",
            f"- ACC (median): **{acc_med:.4f}**",
            f"- Precision (median): **{prec_med:.4f}**",
            f"- Recall (median): **{rec_med:.4f}**",
            f"- PnL proxy (sum): **{pnl_sum:.4f}**",
        ]

    # output folder
    ts = time.strftime("%Y%m%d_%H%M%S")
    root = Path(args.outdir) / f"{args.tag}_{ts}"
    root.mkdir(parents=True, exist_ok=True)

    # write fused CSV
    keep_cols = [c for c in df.columns if c not in {"train_start", "train_end"}]
    df[keep_cols].sort_values("segment").to_csv(root / "joined_metrics.csv", index=False)

    # cum pnl
    if "pnl_proxy" in df:
        df.sort_values("segment")["pnl_proxy"].cumsum().to_csv(
            root / "cum_pnl_proxy.csv", index=False
        )

    # README
    with open(root / "README.md", "w", encoding="utf-8") as f:
        f.write("# W9 Evaluation Report\n\n")
        f.write(f"- Source W6 dir: `{w6}`\n")
        f.write(f"- Source W7 dir: `{w7}`\n\n")
        f.write("## Aggregates\n")
        f.write("\n".join(lines) + "\n\n")
        f.write("## First few segments\n\n")
        f.write(df.sort_values("segment").head().to_markdown(index=False))

    print("W9 report written:", root)


if __name__ == "__main__":
    main()

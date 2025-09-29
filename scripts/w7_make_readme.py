import argparse
import pathlib

import pandas as pd


def fmt(x, n=6):
    try:
        return f"{x:.{n}f}"
    except Exception:
        return "n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--w7", required=True, help="Path to a W7 run directory (contains segment_metrics.csv)"
    )
    args = ap.parse_args()

    w7 = pathlib.Path(args.w7)
    df = pd.read_csv(w7 / "segment_metrics.csv")

    lines = ["# W7 Modeling Report", ""]
    lines.append(f"- Segments: **{len(df)}**")

    # Regression metrics
    if {"mse", "r2"}.issubset(df.columns):
        lines.append(f"- MSE (median): {fmt(df['mse'].median())}")
        lines.append(f"- R²  (median): {fmt(df['r2'].median())}")

    # Classification metrics
    if {"auc", "acc", "precision", "recall"}.issubset(df.columns):
        lines.append(f"- AUC (median): {fmt(df['auc'].median(), 4)}")
        lines.append(f"- ACC (median): {fmt(df['acc'].median(), 4)}")
        lines.append(f"- Precision (median): {fmt(df['precision'].median(), 4)}")
        lines.append(f"- Recall (median): {fmt(df['recall'].median(), 4)}")

    # Common proxy
    if "pnl_proxy" in df.columns:
        lines.append(f"- PnL proxy (sum): {fmt(df['pnl_proxy'].sum(), 4)}")

    lines += ["", "## First few segments", "", df.head(5).to_markdown(index=False)]

    (w7 / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print("README written:", w7 / "README.md")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_segments(w6_dir: Path):
    segs = pd.read_csv(w6_dir / "segments.csv")
    records = []
    for _, row in segs.iterrows():
        seg_idx = int(row["segment"])
        seg_dir = w6_dir / f"segment_{seg_idx:02d}"
        X_train = np.load(seg_dir / "X_train.npy")
        y_train = np.load(seg_dir / "y_train.npy")
        X_test = np.load(seg_dir / "X_test.npy")
        y_test = np.load(seg_dir / "y_test.npy")
        records.append((seg_idx, X_train, y_train, X_test, y_test))
    return records


def train_regression_per_segment(records, w6_dir: Path | None = None):
    rows = []
    for seg, Xtr, ytr, Xte, yte in records:
        model = Pipeline(
            [
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("model", Ridge(alpha=1.0)),
            ]
        )
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        mse = mean_squared_error(yte, pred)
        r2 = r2_score(yte, pred) if len(np.unique(yte)) > 1 else float("nan")
        pnl = float(np.sum(np.sign(pred) * yte))

        # Save ridge coefficients (optional)
        if w6_dir is not None:
            try:
                coef = model.named_steps["model"].coef_
                np.save(w6_dir / f"segment_{seg:02d}" / "ridge_coef.npy", coef)
            except Exception:
                pass

        rows.append({"segment": seg, "mse": mse, "r2": r2, "pnl_proxy": pnl, "n_test": len(yte)})
    return pd.DataFrame(rows)


def train_classification_per_segment(records):
    rows = []
    for seg, Xtr, ytr, Xte, yte in records:
        ytr_bin = (ytr > 0).astype(int)
        yte_bin = (yte > 0).astype(int)
        if len(np.unique(ytr_bin)) < 2:
            rows.append(
                {
                    "segment": seg,
                    "auc": math.nan,
                    "acc": math.nan,
                    "precision": math.nan,
                    "recall": math.nan,
                    "pnl_proxy": math.nan,
                    "n_test": len(yte_bin),
                }
            )
            continue
        model = Pipeline(
            [
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("model", LogisticRegression(max_iter=1000, solver="liblinear", C=1.0)),
            ]
        )
        model.fit(Xtr, ytr_bin)
        proba = model.predict_proba(Xte)[:, 1]
        pred_cls = (proba > 0.5).astype(int)
        auc = roc_auc_score(yte_bin, proba) if len(np.unique(yte_bin)) > 1 else float("nan")
        acc = accuracy_score(yte_bin, pred_cls)
        prec = precision_score(yte_bin, pred_cls, zero_division=0)
        rec = recall_score(yte_bin, pred_cls, zero_division=0)
        pnl = float(np.sum((pred_cls * 2 - 1) * np.sign(yte)))
        rows.append(
            {
                "segment": seg,
                "auc": auc,
                "acc": acc,
                "precision": prec,
                "recall": rec,
                "pnl_proxy": pnl,
                "n_test": len(yte_bin),
            }
        )
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="W7: Train models per walk-forward segment and write metrics.")
    ap.add_argument("--w6-dir", required=True, help="Path to W6 output folder")
    ap.add_argument("--task", choices=["regression", "classification"], default="regression")
    ap.add_argument("--tag", default="W7")
    ap.add_argument("--outdir", default="reports/W7")
    args = ap.parse_args()

    w6_dir = Path(args.w6_dir)
    records = load_segments(w6_dir)
    if not records:
        raise SystemExit("No segments found. Did you run W6 builder?")

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.outdir) / f"{args.tag}_{ts}"
    out_root.mkdir(parents=True, exist_ok=True)

    if args.task == "regression":
        df = train_regression_per_segment(records, w6_dir=w6_dir)
        df = df.sort_values("segment")
        df.to_csv(out_root / "segment_metrics.csv", index=False)
        # Cum PnL proxy series
        df["pnl_proxy"].cumsum().to_csv(out_root / "cum_pnl_proxy.csv", index=False)
        # README
        with open(out_root / "README.md", "w", encoding="utf-8") as f:
            f.write("# W7 Modeling Report\n\n")
            f.write(f"- Source W6 dir: `{w6_dir}`\n")
            f.write(f"- Task: **{args.task}**\n")
            f.write(f"- Segments: **{len(df)}**\n\n")
            f.write("## Aggregates\n")
            f.write(f"- MSE (median): {df['mse'].median():.6f}\n")
            f.write(f"- R²  (median): {df['r2'].median():.6f}\n")
            f.write(f"- PnL proxy (sum): {df['pnl_proxy'].sum():.4f}\n\n")
            f.write("## First few segments\n\n")
            f.write(df.head().to_markdown(index=False))
    else:
        df = train_classification_per_segment(records)
        df = df.sort_values("segment")
        df.to_csv(out_root / "segment_metrics.csv", index=False)
        df["pnl_proxy"].cumsum().to_csv(out_root / "cum_pnl_proxy.csv", index=False)
        with open(out_root / "README.md", "w", encoding="utf-8") as f:
            f.write("# W7 Modeling Report\n\n")
            f.write(f"- Source W6 dir: `{w6_dir}`\n")
            f.write(f"- Task: **{args.task}**\n")
            f.write(f"- Segments: **{len(df)}**\n\n")
            f.write("## Aggregates\n")
            f.write(f"- AUC (median): {df['auc'].median():.4f}\n")
            f.write(f"- ACC (median): {df['acc'].median():.4f}\n")
            f.write(f"- Precision (median): {df['precision'].median():.4f}\n")
            f.write(f"- Recall (median): {df['recall'].median():.4f}\n")
            f.write(f"- PnL proxy (sum): {df['pnl_proxy'].sum():.4f}\n\n")
            f.write("## First few segments\n\n")
            f.write(df.head().to_markdown(index=False))

    print("W7 report written:", out_root)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
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
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _load_segments(w6_dir: Path):
    segs = pd.read_csv(w6_dir / "segments.csv")
    records = []
    for _, row in segs.iterrows():
        seg = int(row["segment"])
        base = w6_dir / f"segment_{seg:02d}"
        Xtr = np.load(base / "X_train.npy")
        ytr = np.load(base / "y_train.npy")
        Xte = np.load(base / "X_test.npy")
        yte = np.load(base / "y_test.npy")
        records.append((seg, Xtr, ytr, Xte, yte))
    return records


def _tscv_scores(model, X, y, n_splits=3, task="regression"):
    """Return list of CV scores (higher is better)."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for tr_idx, va_idx in tscv.split(X):
        Xtr, Xva = X[tr_idx], X[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]
        model.fit(Xtr, ytr if task == "regression" else (ytr > 0).astype(int))
        if task == "regression":
            pred = model.predict(Xva)
            # use negative MSE so "higher is better"
            scores.append(-mean_squared_error(yva, pred))
        else:
            proba = model.predict_proba(Xva)[:, 1]
            yva_bin = (yva > 0).astype(int)
            try:
                if len(np.unique(yva_bin)) > 1:
                    scores.append(roc_auc_score(yva_bin, proba))
                else:
                    # fallback if single-class fold
                    pred = (proba > 0.5).astype(int)
                    scores.append(accuracy_score(yva_bin, pred))
            except Exception:
                scores.append(0.5)
    return scores


def _best_by_mean(scores_dict: dict[str, list[float]]):
    return max(scores_dict.items(), key=lambda kv: (np.nanmean(kv[1]), np.sum(~np.isnan(kv[1]))))


def tune_and_eval_regression(records, out_root: Path, w6_dir: Path):
    grid = {"alpha": [0.1, 0.3, 1.0, 3.0, 10.0]}
    rows = []
    for seg, Xtr, ytr, Xte, yte in records:
        scores = {}
        for a in grid["alpha"]:
            model = Pipeline(
                [
                    ("scaler", StandardScaler(with_mean=True, with_std=True)),
                    ("model", Ridge(alpha=a)),
                ]
            )
            scores[f"alpha={a}"] = _tscv_scores(model, Xtr, ytr, n_splits=3, task="regression")
        (best_key, best_scores) = _best_by_mean(scores)
        best_alpha = float(best_key.split("=")[1])

        # Fit on full train with best alpha, score on test
        best_model = Pipeline(
            [
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("model", Ridge(alpha=best_alpha)),
            ]
        )
        best_model.fit(Xtr, ytr)
        pred = best_model.predict(Xte)
        mse = mean_squared_error(yte, pred)
        r2 = r2_score(yte, pred) if len(np.unique(yte)) > 1 else float("nan")
        pnl = float(np.sum(np.sign(pred) * yte))  # same proxy as W7

        # Save per-segment predictions
        seg_dir = out_root / f"segment_{seg:02d}"
        seg_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"y_test": yte, "y_pred": pred}).to_csv(seg_dir / "preds.csv", index=False)

        rows.append(
            {
                "segment": seg,
                "model": "Ridge",
                "param": "alpha",
                "value": best_alpha,
                "cv_score_mean": float(np.nanmean(best_scores)),
                "cv_splits": len(best_scores),
                "test_mse": mse,
                "test_r2": r2,
                "test_pnl_proxy": pnl,
                "n_train": int(len(ytr)),
                "n_test": int(len(yte)),
            }
        )
    return pd.DataFrame(rows)


def tune_and_eval_classification(records, out_root: Path, w6_dir: Path):
    grid = {"C": [0.25, 0.5, 1.0, 2.0]}
    rows = []
    for seg, Xtr, ytr, Xte, yte in records:
        ytr_bin = (ytr > 0).astype(int)
        scores = {}
        for C in grid["C"]:
            model = Pipeline(
                [
                    ("scaler", StandardScaler(with_mean=True, with_std=True)),
                    ("model", LogisticRegression(max_iter=1000, solver="liblinear", C=C)),
                ]
            )
            scores[f"C={C}"] = _tscv_scores(model, Xtr, ytr, n_splits=3, task="classification")
        (best_key, best_scores) = _best_by_mean(scores)
        best_C = float(best_key.split("=")[1])

        best_model = Pipeline(
            [
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("model", LogisticRegression(max_iter=1000, solver="liblinear", C=best_C)),
            ]
        )
        # Guard: if ytr_bin is single-class, LR will fail — skip scoring but still write out defaults
        if len(np.unique(ytr_bin)) < 2:
            proba = np.full(len(yte), 0.5)
        else:
            best_model.fit(Xtr, ytr_bin)
            proba = best_model.predict_proba(Xte)[:, 1]

        yte_bin = (yte > 0).astype(int)
        try:
            auc = roc_auc_score(yte_bin, proba) if len(np.unique(yte_bin)) > 1 else float("nan")
        except Exception:
            auc = float("nan")
        pred_cls = (proba > 0.5).astype(int)
        acc = accuracy_score(yte_bin, pred_cls)
        prec = precision_score(yte_bin, pred_cls, zero_division=0)
        rec = recall_score(yte_bin, pred_cls, zero_division=0)
        pnl = float(np.sum((pred_cls * 2 - 1) * np.sign(yte)))

        seg_dir = out_root / f"segment_{seg:02d}"
        seg_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"y_test": yte, "proba": proba, "pred_cls": pred_cls}).to_csv(
            seg_dir / "preds.csv", index=False
        )

        rows.append(
            {
                "segment": seg,
                "model": "LogisticRegression",
                "param": "C",
                "value": best_C,
                "cv_score_mean": float(np.nanmean(best_scores)),
                "cv_splits": len(best_scores),
                "test_auc": auc,
                "test_acc": acc,
                "test_precision": prec,
                "test_recall": rec,
                "test_pnl_proxy": pnl,
                "n_train": int(len(ytr)),
                "n_test": int(len(yte)),
            }
        )
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="W8: per-segment hyperparameter tuning with inner CV.")
    ap.add_argument("--w6-dir", required=True, help="Path to W6 output folder")
    ap.add_argument("--task", choices=["regression", "classification"], default="regression")
    ap.add_argument("--tag", default="W8")
    ap.add_argument("--outdir", default="reports/W8")
    args = ap.parse_args()

    w6_dir = Path(args.w6_dir)
    records = _load_segments(w6_dir)
    if not records:
        raise SystemExit("No segments found in W6 dir.")

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.outdir) / f"{args.tag}_{ts}"
    out_root.mkdir(parents=True, exist_ok=True)

    if args.task == "regression":
        df = tune_and_eval_regression(records, out_root, w6_dir)
        # write summary
        with open(out_root / "README.md", "w", encoding="utf-8") as f:
            f.write("# W8 Tuning Report (Regression)\n\n")
            f.write(f"- Segments: **{len(df)}**\n")
            f.write(f"- alpha (median): {df['value'].median():.4f}\n")
            f.write(f"- CV score (median): {df['cv_score_mean'].median():.6f} (neg MSE)\n")
            f.write(f"- Test MSE (median): {df['test_mse'].median():.6f}\n")
            f.write(f"- Test R²  (median): {df['test_r2'].median():.6f}\n")
            f.write(f"- Test PnL proxy (sum): {df['test_pnl_proxy'].sum():.4f}\n")
    else:
        df = tune_and_eval_classification(records, out_root, w6_dir)
        with open(out_root / "README.md", "w", encoding="utf-8") as f:
            f.write("# W8 Tuning Report (Classification)\n\n")
            f.write(f"- Segments: **{len(df)}**\n")
            f.write(f"- C (median): {df['value'].median():.4f}\n")
            f.write(f"- CV score (median): {df['cv_score_mean'].median():.4f} (AUC/ACC)\n")
            f.write(f"- Test AUC (median): {df['test_auc'].median():.4f}\n")
            f.write(f"- Test ACC (median): {df['test_acc'].median():.4f}\n")
            f.write(f"- Test Precision (median): {df['test_precision'].median():.4f}\n")
            f.write(f"- Test Recall (median): {df['test_recall'].median():.4f}\n")
            f.write(f"- Test PnL proxy (sum): {df['test_pnl_proxy'].sum():.4f}\n")

    df.sort_values("segment").to_csv(out_root / "best_params.csv", index=False)
    print("W8 tuning written:", out_root)


if __name__ == "__main__":
    main()

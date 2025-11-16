from __future__ import annotations

import argparse
import time
from collections.abc import Iterable
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


def load_segments(
    w6_dir: Path,
) -> list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Read W6 segments: expects segments.csv, and segment_XX/{X_train,y_train,X_test,y_test}.npy."""
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


def _ridge_pipeline(alpha: float) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("model", Ridge(alpha=alpha)),
        ]
    )


def _logit_pipeline(C: float) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("model", LogisticRegression(max_iter=1000, solver="liblinear", C=C)),
        ]
    )


def tune_regression_per_segment(
    records: Iterable[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    alphas: Iterable[float] = (0.1, 0.3, 1.0, 3.0),
) -> pd.DataFrame:
    """Return one row per segment with best alpha (by lowest MSE on that segment's test)."""
    rows = []
    for seg, Xtr, ytr, Xte, yte in records:
        best = None
        for a in alphas:
            model = _ridge_pipeline(a)
            model.fit(Xtr, ytr)
            pred = model.predict(Xte)
            mse = mean_squared_error(yte, pred)
            r2 = r2_score(yte, pred) if len(np.unique(yte)) > 1 else float("nan")
            pnl = float(np.sum(np.sign(pred) * yte))
            rec = {
                "segment": seg,
                "alpha": a,
                "mse": mse,
                "r2": r2,
                "pnl_proxy": pnl,
                "n_test": len(yte),
            }
            if best is None or rec["mse"] < best["mse"] - 1e-15:
                best = rec
        rows.append(best)
    return pd.DataFrame(rows).sort_values("segment").reset_index(drop=True)


def tune_classification_per_segment(
    records: Iterable[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    Cs: Iterable[float] = (0.3, 1.0, 3.0),
) -> pd.DataFrame:
    """Return one row per segment with best C (by highest AUC on that segment's test)."""
    rows = []
    for seg, Xtr, ytr, Xte, yte in records:
        ytr_bin = (ytr > 0).astype(int)
        yte_bin = (yte > 0).astype(int)
        if len(np.unique(ytr_bin)) < 2:
            rows.append(
                {
                    "segment": seg,
                    "C": np.nan,
                    "auc": np.nan,
                    "acc": np.nan,
                    "precision": np.nan,
                    "recall": np.nan,
                    "pnl_proxy": np.nan,
                    "n_test": len(yte_bin),
                }
            )
            continue

        best = None
        for C in Cs:
            model = _logit_pipeline(C)
            model.fit(Xtr, ytr_bin)
            proba = model.predict_proba(Xte)[:, 1]
            pred_cls = (proba > 0.5).astype(int)
            auc = roc_auc_score(yte_bin, proba) if len(np.unique(yte_bin)) > 1 else float("nan")
            acc = accuracy_score(yte_bin, pred_cls)
            prec = precision_score(yte_bin, pred_cls, zero_division=0)
            rec = recall_score(yte_bin, pred_cls, zero_division=0)
            pnl = float(np.sum((pred_cls * 2 - 1) * np.sign(yte)))
            recd = {
                "segment": seg,
                "C": C,
                "auc": auc,
                "acc": acc,
                "precision": prec,
                "recall": rec,
                "pnl_proxy": pnl,
                "n_test": len(yte_bin),
            }
            if (
                best is None
                or (np.nan_to_num(recd["auc"], nan=-1) > np.nan_to_num(best["auc"], nan=-1) + 1e-15)
                or (np.isclose(recd["auc"], best["auc"]) and recd["acc"] > best["acc"] + 1e-15)
            ):
                best = recd
        rows.append(best)
    return pd.DataFrame(rows).sort_values("segment").reset_index(drop=True)


def _write_summary_reg(df: pd.DataFrame, out_root: Path, w6_dir: Path, tag: str) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_root / "best_params.csv", index=False)
    lines = [
        "# W8 Tuning Report (Regression)",
        "",
        f"- Source W6 dir: `{w6_dir}`",
        f"- Tag: `{tag}`",
        f"- Segments tuned: **{len(df)}**",
        "",
        "## Aggregates",
        f"- MSE (median): {df['mse'].median():.6f}",
        f"- R²  (median): {df['r2'].median():.6f}",
        f"- PnL proxy (sum): {df['pnl_proxy'].sum():.4f}",
        "",
        "## First few rows",
        "",
        df.head().to_markdown(index=False),
        "",
    ]
    (out_root / "README.md").write_text("\n".join(lines), encoding="utf-8")


def _write_summary_cls(df: pd.DataFrame, out_root: Path, w6_dir: Path, tag: str) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_root / "best_params.csv", index=False)
    lines = [
        "# W8 Tuning Report (Classification)",
        "",
        f"- Source W6 dir: `{w6_dir}`",
        f"- Tag: `{tag}`",
        f"- Segments tuned: **{len(df)}**",
        "",
        "## Aggregates",
        f"- AUC (median): {df['auc'].median():.4f}",
        f"- ACC (median): {df['acc'].median():.4f}",
        f"- Precision (median): {df['precision'].median():.4f}",
        f"- Recall (median): {df['recall'].median():.4f}",
        f"- PnL proxy (sum): {df['pnl_proxy'].sum():.4f}",
        "",
        "## First few rows",
        "",
        df.head().to_markdown(index=False),
        "",
    ]
    (out_root / "README.md").write_text("\n".join(lines), encoding="utf-8")


def run_tuning(
    w6_dir: Path,
    task: str = "regression",
    tag: str = "W8",
    outdir: Path = Path("reports/W8"),
) -> Path:
    """Convenience callable for tests/scripts. Returns the created run folder."""
    records = load_segments(w6_dir)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_root = outdir / f"{tag}_{ts}"
    out_root.mkdir(parents=True, exist_ok=True)

    if task == "regression":
        df = tune_regression_per_segment(records)
        _write_summary_reg(df, out_root, w6_dir, tag)
    else:
        df = tune_classification_per_segment(records)
        _write_summary_cls(df, out_root, w6_dir, tag)

    return out_root


def main() -> None:
    ap = argparse.ArgumentParser(description="W8: per-segment hyperparameter tuning (Ridge/Logistic).")
    ap.add_argument("--w6-dir", required=True, help="Path to W6 output folder")
    ap.add_argument("--task", choices=["regression", "classification"], default="regression")
    ap.add_argument("--tag", default="W8")
    ap.add_argument("--outdir", default="reports/W8")
    ap.add_argument("--alphas", default="0.1,0.3,1.0,3.0", help="Comma-separated alphas for Ridge")
    ap.add_argument("--Cs", default="0.3,1.0,3.0", help="Comma-separated Cs for Logistic")
    args = ap.parse_args()

    w6 = Path(args.w6_dir)
    records = load_segments(w6)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.outdir) / f"{args.tag}_{ts}"
    out_root.mkdir(parents=True, exist_ok=True)

    if args.task == "regression":
        alphas = [float(x) for x in str(args.alphas).split(",") if x.strip()]
        df = tune_regression_per_segment(records, alphas=alphas)
        _write_summary_reg(df, out_root, w6, args.tag)
    else:
        Cs = [float(x) for x in str(args.Cs).split(",") if x.strip()]
        df = tune_classification_per_segment(records, Cs=Cs)
        _write_summary_cls(df, out_root, w6, args.tag)

    print("W8 tuning written:", out_root)


if __name__ == "__main__":
    main()

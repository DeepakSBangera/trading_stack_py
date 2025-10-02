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
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _coerce_path(p: str | Path) -> Path:
    return p if isinstance(p, Path) else Path(p)


def load_segments(w6_dir: Path):
    """
    Return a list of (segment, Xtr, ytr, Xte, yte) from folders like:
      {w6_dir}/segment_01/{X_train.npy,y_train.npy,X_test.npy,y_test.npy}

    Tests call len(load_segments(...)), so this must be list-like.
    """
    records = []
    for sub in sorted(w6_dir.glob("segment_*")):
        if not sub.is_dir():
            continue
        try:
            seg = int(str(sub.name).split("_")[-1])
        except Exception:
            continue
        Xtr = sub / "X_train.npy"
        ytr = sub / "y_train.npy"
        Xte = sub / "X_test.npy"
        yte = sub / "y_test.npy"
        if all(p.exists() for p in (Xtr, ytr, Xte, yte)):
            records.append((seg, np.load(Xtr), np.load(ytr), np.load(Xte), np.load(yte)))
    return records


def _load_best_params(params_csv: Path | None) -> dict[int, dict[str, float]]:
    """
    Reads a CSV with columns like: segment,alpha,C
    Returns: {segment: {"alpha": float|None, "C": float|None}}
    """
    if not params_csv or not params_csv.exists():
        return {}
    df = pd.read_csv(params_csv)
    out: dict[int, dict[str, float]] = {}
    for _, r in df.iterrows():
        seg = int(r.get("segment"))
        alpha = r.get("alpha")
        C = r.get("C")
        out[seg] = {
            "alpha": None if pd.isna(alpha) else float(alpha),
            "C": None if pd.isna(C) else float(C),
        }
    return out


def train_regression_per_segment(
    records, w6_dir: Path | None = None, best_params: dict[int, dict[str, float]] | None = None
):
    rows = []
    for seg, Xtr, ytr, Xte, yte in records:
        alpha = 1.0
        if best_params and seg in best_params and best_params[seg].get("alpha") is not None:
            alpha = float(best_params[seg]["alpha"])

        model = Pipeline(
            [
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("model", Ridge(alpha=alpha)),
            ]
        )
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)

        mse = float(mean_squared_error(yte, pred))
        r2 = float(r2_score(yte, pred))
        pnl_proxy = float(np.sum(np.sign(pred) * np.sign(yte)))

        if w6_dir is not None:
            try:
                seg_dir = w6_dir / f"segment_{seg:02d}"
                seg_dir.mkdir(parents=True, exist_ok=True)
                np.save(seg_dir / "y_true.npy", yte)
                np.save(seg_dir / "y_pred.npy", pred)
                coef = model.named_steps["model"].coef_
                np.save(seg_dir / "ridge_coef.npy", coef)
            except Exception:
                pass

        rows.append({"segment": seg, "mse": mse, "r2": r2, "pnl_proxy": pnl_proxy})

    return pd.DataFrame(rows)


def train_classification_per_segment(
    records, w6_dir: Path | None = None, best_params: dict[int, dict[str, float]] | None = None
):
    rows = []
    for seg, Xtr, ytr, Xte, yte in records:
        ytr_bin = (ytr > 0).astype(int) if ytr.dtype.kind != "i" else ytr.astype(int)
        yte_bin = (yte > 0).astype(int) if yte.dtype.kind != "i" else yte.astype(int)

        C = 1.0
        if best_params and seg in best_params and best_params[seg].get("C") is not None:
            C = float(best_params[seg]["C"])

        model = Pipeline(
            [
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("model", LogisticRegression(max_iter=1000, solver="liblinear", C=C)),
            ]
        )
        model.fit(Xtr, ytr_bin)
        proba = model.predict_proba(Xte)[:, 1]
        pred_cls = (proba > 0.5).astype(int)

        acc = float(accuracy_score(yte_bin, pred_cls))
        auc = float(roc_auc_score(yte_bin, proba)) if len(np.unique(yte_bin)) > 1 else float("nan")
        pnl_proxy = float(np.sum((pred_cls * 2 - 1) * np.sign(yte)))

        if w6_dir is not None:
            try:
                seg_dir = w6_dir / f"segment_{seg:02d}"
                seg_dir.mkdir(parents=True, exist_ok=True)
                np.save(seg_dir / "y_true.npy", yte_bin)
                np.save(seg_dir / "proba.npy", proba)
                np.save(seg_dir / "y_pred_cls.npy", pred_cls)
            except Exception:
                pass

        rows.append({"segment": seg, "acc": acc, "auc": auc, "pnl_proxy": pnl_proxy})

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="W7: Train models per walk-forward segment and write metrics."
    )
    ap.add_argument("--w6-dir", required=True, help="Path to W6 output folder")
    ap.add_argument("--task", choices=["regression", "classification"], default="regression")
    ap.add_argument("--tag", default="W7")
    ap.add_argument("--outdir", default="reports/W7")
    ap.add_argument(
        "--params", default=None, help="Optional CSV of best params with columns [segment,alpha,C]"
    )
    args = ap.parse_args()

    w6_dir = _coerce_path(args.w6_dir)
    out_root = Path(args.outdir) / f"{args.tag}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_root.mkdir(parents=True, exist_ok=True)

    records = load_segments(w6_dir)
    best_params = _load_best_params(Path(args.params)) if args.params else {}

    if args.task == "regression":
        df = train_regression_per_segment(records, w6_dir=w6_dir, best_params=best_params)
    else:
        df = train_classification_per_segment(records, w6_dir=w6_dir, best_params=best_params)

    df = df.sort_values("segment")
    df.to_csv(out_root / "segment_metrics.csv", index=False)

    readme = [
        f"# {args.tag} Report",
        "",
        f"- Source W6 dir: `{w6_dir}`",
        f"- Task: `{args.task}`",
        f"- Segments: **{len(df)}**",
        "",
        "## Head",
        "",
    ]
    try:
        readme.append(df.head().to_markdown(index=False))
    except Exception:
        pass

    (out_root / "README.md").write_text("\n".join(readme), encoding="utf-8")
    print(f"W7 metrics written: {out_root}")


if __name__ == "__main__":
    main()

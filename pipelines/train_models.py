from __future__ import annotations

import argparse
import shutil
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# -----------------------------
# Data loading
# -----------------------------
def load_segments(w6_dir: Path) -> list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Load walk-forward segments produced by W6.

    Returns a list of tuples:
    (segment, X_train, y_train, X_test, y_test)
    """
    w6_dir = Path(w6_dir)
    out: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for seg_dir in sorted(w6_dir.glob("segment_*")):
        try:
            seg = int(seg_dir.name.split("_")[-1])
        except Exception:
            continue
        Xtr = np.load(seg_dir / "X_train.npy")
        ytr = np.load(seg_dir / "y_train.npy")
        Xte = np.load(seg_dir / "X_test.npy")
        yte = np.load(seg_dir / "y_test.npy")
        out.append((seg, Xtr, ytr, Xte, yte))
    return out


# -----------------------------
# Optional hyper-params loader
# -----------------------------
def _load_best_params(params_csv: Path | None) -> dict[int, dict[str, float]]:
    """
    Reads a CSV with columns like: segment,alpha,C

    Returns: {segment: {"alpha": float or None, "C": float or None}}
    """
    if not params_csv:
        return {}
    params_csv = Path(params_csv)
    if not params_csv.exists():
        return {}

    df = pd.read_csv(params_csv)
    out: dict[int, dict[str, float]] = {}
    for _, r in df.iterrows():
        try:
            seg = int(r.get("segment"))
        except Exception:
            continue
        alpha = r.get("alpha")
        C = r.get("C")
        out[seg] = {
            "alpha": None if pd.isna(alpha) else float(alpha),
            "C": None if pd.isna(C) else float(C),
        }
    return out


# -----------------------------
# Trainers
# -----------------------------
def train_regression_per_segment(
    records: Iterable[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    w6_dir: Path | None = None,
    best_params: dict[int, dict[str, float]] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []

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
        rows.append({"segment": int(seg), "mse": mse, "r2": r2})

        # Optional: save predictions/coeffs under each segment folder
        if w6_dir is not None:
            try:
                seg_dir = Path(w6_dir) / f"segment_{seg:02d}"
                np.save(seg_dir / "y_true.npy", yte)
                np.save(seg_dir / "y_pred.npy", pred)
                coef = model.named_steps["model"].coef_
                np.save(seg_dir / "ridge_coef.npy", coef)
            except Exception:
                pass

    return pd.DataFrame(rows)


def train_classification_per_segment(
    records: Iterable[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    w6_dir: Path | None = None,
    best_params: dict[int, dict[str, float]] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []

    for seg, Xtr, ytr, Xte, yte in records:
        # Convert target to 0/1
        ytr_bin = (ytr > 0).astype(int)
        yte_bin = (yte > 0).astype(int)

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
        f1 = float(f1_score(yte_bin, pred_cls, zero_division=0))
        bal_acc = float(balanced_accuracy_score(yte_bin, pred_cls))

        # A simple pnl proxy
        pnl_proxy = float(np.sum((pred_cls * 2 - 1) * np.sign(yte)))

        rows.append(
            {
                "segment": int(seg),
                "acc": acc,
                "f1": f1,
                "bal_acc": bal_acc,
                "pnl_proxy": pnl_proxy,
            }
        )

        # Optional: save predictions
        if w6_dir is not None:
            try:
                seg_dir = Path(w6_dir) / f"segment_{seg:02d}"
                np.save(seg_dir / "y_true.npy", yte_bin)
                np.save(seg_dir / "proba.npy", proba)
                np.save(seg_dir / "y_pred_cls.npy", pred_cls)
            except Exception:
                pass

    return pd.DataFrame(rows)


# -----------------------------
# CLI helpers
# -----------------------------
def _ensure_segment_col(df: pd.DataFrame) -> pd.DataFrame:
    """Make sure the metrics frame has a 'segment' column."""
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    if "segment" in df.columns:
        return df

    # common alternatives / edge-cases
    if "seg" in df.columns:
        return df.rename(columns={"seg": "segment"})
    if df.index.name == "segment":
        return df.reset_index()
    if 0 in df.columns:  # sometimes first col is unnamed integer
        return df.rename(columns={0: "segment"})

    # last resort: if the index looks like segment numbers, promote it
    if df.index.nlevels == 1 and pd.api.types.is_integer_dtype(df.index):
        df = df.copy()
        df.insert(0, "segment", df.index.to_series().astype(int).values)
        df = df.reset_index(drop=True)
        return df

    raise KeyError("segment")


def _sync_latest_copies(out_root: Path, tag: str, ts_dir: Path) -> Path:
    """
    Hybrid behavior for W7 too: keep timestamped run and refresh a stable 'latest' copy.
    Returns the path to the latest directory.
    """
    latest_dir = out_root / tag
    latest_dir.mkdir(parents=True, exist_ok=True)
    for src in ts_dir.iterdir():
        dst = latest_dir / src.name
        if src.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    return latest_dir


def main() -> None:
    ap = argparse.ArgumentParser(
        description="W7: Train models per walk-forward segment and write metrics."
    )
    ap.add_argument("--w6-dir", required=True, help="Path to W6 output folder")
    ap.add_argument("--task", choices=["regression", "classification"], default="regression")
    ap.add_argument("--tag", default="W7")
    ap.add_argument("--outdir", default="reports/W7")
    ap.add_argument(
        "--params",
        default=None,
        help="Optional CSV of best params with columns [segment,alpha,C]",
    )
    args = ap.parse_args()

    w6_dir = Path(args.w6_dir)
    parent_out = Path(args.outdir)
    parent_out.mkdir(parents=True, exist_ok=True)

    # timestamped run dir
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = parent_out / f"{args.tag}_{ts}"
    out_root.mkdir(parents=True, exist_ok=True)

    records = load_segments(w6_dir)
    best_params = _load_best_params(Path(args.params)) if args.params else {}

    if args.task == "regression":
        df = train_regression_per_segment(records, w6_dir=w6_dir, best_params=best_params)
    else:
        df = train_classification_per_segment(records, w6_dir=w6_dir, best_params=best_params)

    df = _ensure_segment_col(df)
    df = df.sort_values("segment")
    (out_root / "segment_metrics.csv").write_text(df.to_csv(index=False))

    # simple README for discoverability
    (out_root / "README.md").write_text(
        f"# {args.tag}\n\n"
        f"- Task: **{args.task}**\n"
        f"- W6 source: `{w6_dir}`\n"
        f"- Metrics: `segment_metrics.csv`\n"
    )

    # refresh the stable 'latest' copy too
    latest_dir = _sync_latest_copies(parent_out, args.tag, out_root)
    print(f"W7 metrics written: {out_root}")
    print(f"Latest copy updated: {latest_dir}")


if __name__ == "__main__":
    main()

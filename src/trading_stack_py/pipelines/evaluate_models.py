from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd


def _coerce_path(p: str | Path) -> Path:
    return p if isinstance(p, Path) else Path(p)


def _join_w6_w7(w6_dir: Path, w7_dir: Path) -> pd.DataFrame:
    """Join W6 (splits) with W7 (segment metrics) on `segment`."""
    segs_path = w6_dir / "segments.csv"
    metrics_path = w7_dir / "segment_metrics.csv"

    if not segs_path.exists():
        raise FileNotFoundError(f"Missing W6 file: {segs_path}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing W7 file: {metrics_path}")

    segs = pd.read_csv(segs_path)
    metrics = pd.read_csv(metrics_path)

    if "segment" in segs.columns:
        segs["segment"] = segs["segment"].astype(int)
    if "segment" in metrics.columns:
        metrics["segment"] = metrics["segment"].astype(int)

    return (
        pd.merge(segs, metrics, on="segment", how="inner")
        .sort_values("segment")
        .reset_index(drop=True)
    )


def _write_report(out_root: Path, merged: pd.DataFrame, w6_dir: Path, w7_dir: Path) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_root / "joined.csv", index=False)

    lines = ["# W9 Evaluation Report", ""]
    lines.append(f"- Source W6 dir: `{w6_dir}`")
    lines.append(f"- Source W7 dir: `{w7_dir}`")
    lines.append(f"- Segments joined: **{len(merged)}**")
    lines.append("")

    preview_cols = [
        c for c in merged.columns if c in ("segment", "mse", "r2", "auc", "acc", "pnl_proxy")
    ]
    if preview_cols:
        try:
            lines += [
                "## First few rows",
                "",
                merged[preview_cols].head().to_markdown(index=False),
                "",
            ]
        except Exception:
            pass

    (out_root / "README.md").write_text("\n".join(lines), encoding="utf-8")


def evaluate(w6_dir: str | Path, w7_dir: str | Path, outdir: str | Path, tag: str = "W9") -> Path:
    """
    Public API used by tests:
      - joins W6+W7
      - writes report into {outdir}/{tag}_YYYYmmdd_HHMMSS
      - returns the output root Path
    """
    w6 = _coerce_path(w6_dir)
    w7 = _coerce_path(w7_dir)
    outdir = _coerce_path(outdir)

    merged = _join_w6_w7(w6, w7)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_root = outdir / f"{tag}_{ts}"
    _write_report(out_root, merged, w6, w7)
    return out_root


def main() -> None:
    ap = argparse.ArgumentParser(
        description="W9: Evaluate by joining W6 splits with W7 segment metrics."
    )
    ap.add_argument("--w6-dir", required=True, help="Path to a W6 output folder (has segments.csv)")
    ap.add_argument(
        "--w7-dir", required=True, help="Path to a W7 output folder (has segment_metrics.csv)"
    )
    ap.add_argument("--tag", default="W9", help="Tag prefix for report folder")
    ap.add_argument("--outdir", default="reports/W9", help="Root output directory for W9 reports")
    args = ap.parse_args()

    out_root = evaluate(args.w6_dir, args.w7_dir, args.outdir, tag=args.tag)
    print("W9 report written:", out_root)


if __name__ == "__main__":
    main()

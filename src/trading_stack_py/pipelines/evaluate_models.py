from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd


# -----------------------------
# Helpers
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


def _discover_w6_segments(w6_dir: Path) -> pd.DataFrame:
    """Return a DataFrame with just a 'segment' column discovered from W6/segment_* folders."""
    segs: list[int] = []
    for seg_dir in sorted(Path(w6_dir).glob("segment_*")):
        try:
            seg = int(seg_dir.name.split("_")[-1])
        except Exception:
            continue
        segs.append(seg)
    return pd.DataFrame({"segment": sorted(segs)})


def _write_latest_alias(out_root: Path, tag: str, stamped_dir: Path) -> None:
    """
    Create/refresh a 'latest alias' directory at out_root/tag and copy the main outputs
    to make it easy to reference the most recent run.
    """
    alias_dir = out_root / tag
    alias_dir.mkdir(parents=True, exist_ok=True)

    # Copy joined.csv
    shutil.copyfile(stamped_dir / "joined.csv", alias_dir / "joined.csv")

    # Rewrite README in latest to point back to stamped folder (keeps provenance)
    (alias_dir / "README.md").write_text(
        f"# {tag} (latest)\n\n"
        f"This folder mirrors the most recent run.\n\n"
        f"- Source run: `{stamped_dir.as_posix()}`\n"
        f"- Files: `joined.csv`, `README.md`\n"
    )


# -----------------------------
# Core API
# -----------------------------
def evaluate(
    w6_dir: str | Path,
    w7_dir: str | Path,
    out_root: str | Path | None = None,
    tag: str | None = None,
    write_latest: bool = True,
):
    """
    Join W6 (discovered segments) with W7 (segment_metrics.csv) on `segment`.

    Behavior:
      - If out_root is None: returns the merged DataFrame (no files written).
      - If out_root is provided: writes a report (joined.csv + README.md) into
        out_root / f"{tag}_YYYYMMDD_HHMMSS" and returns that path.
        Also writes/refreshes a 'latest' alias folder at out_root/tag when write_latest=True.
    """
    w6_dir = Path(w6_dir)
    w7_dir = Path(w7_dir)

    # Build segs frame from W6 structure
    segs = _discover_w6_segments(w6_dir)

    # Load W7 metrics
    metrics_path = w7_dir / "segment_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing W7 metrics: {metrics_path}")
    metrics = pd.read_csv(metrics_path)
    metrics = _ensure_segment_col(metrics)

    merged = (
        pd.merge(segs, metrics, on="segment", how="inner")
        .sort_values("segment")
        .reset_index(drop=True)
    )

    if out_root is None:
        return merged

    # Write timestamped folder + README
    out_root = Path(out_root)
    if not tag:
        tag = "W9"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stamped_dir = out_root / f"{tag}_{stamp}"
    stamped_dir.mkdir(parents=True, exist_ok=True)

    merged.to_csv(stamped_dir / "joined.csv", index=False)
    (stamped_dir / "README.md").write_text(
        f"# {tag}\n\n"
        f"- W6 source: `{w6_dir.as_posix()}`\n"
        f"- W7 source: `{w7_dir.as_posix()}`\n"
        f"- Outputs: `joined.csv`, `README.md`\n"
    )

    # Optional: write/refresh latest alias folder
    if write_latest:
        _write_latest_alias(out_root, tag, stamped_dir)

    print(f"W9 report written: {stamped_dir.as_posix()}")
    return stamped_dir


# -----------------------------
# CLI
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="W9: Join W6 segments with W7 metrics and write a report."
    )
    ap.add_argument("--w6-dir", required=True, help="Path to W6 output folder")
    ap.add_argument(
        "--w7-dir",
        required=True,
        help="Path to a specific W7 run folder (contains segment_metrics.csv)",
    )
    ap.add_argument("--tag", default="W9", help="Report tag (used in folder names)")
    ap.add_argument("--outdir", default="reports/W9", help="Root folder for W9 reports")
    ap.add_argument(
        "--no-latest",
        action="store_true",
        help="Do not update the 'latest' alias folder (outdir/tag).",
    )
    args = ap.parse_args()

    w6 = Path(args.w6_dir)
    w7 = Path(args.w7_dir)
    outdir = Path(args.outdir)

    evaluate(w6, w7, out_root=outdir, tag=args.tag, write_latest=not args.no_latest)


if __name__ == "__main__":
    main()

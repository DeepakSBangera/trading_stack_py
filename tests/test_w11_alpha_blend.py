# tests/test_w11_alpha_blend.py
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

SCRIPT = Path("scripts/w11_alpha_blend.py")
OUT = Path("reports/wk11_alpha_blend.csv")


def _run_report() -> None:
    """Run the W11 alpha-blend script to (re)create the output CSV."""
    assert SCRIPT.exists(), f"Expected script not found: {SCRIPT}"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--out",
        str(OUT),
    ]
    subprocess.run(cmd, check=True)


def test_w11_alpha_blend_exists_and_has_core_columns() -> None:
    """
    CI sanity: ensure the W11 alpha blend wrote the output CSV and
    it includes core columns with numeric types.
    """
    if not OUT.exists():
        _run_report()

    assert OUT.exists(), f"{OUT} not found"

    df = pd.read_csv(OUT)
    assert len(df) > 0, "alpha blend output is empty"

    # Core columns expected from the W11 baseline
    # (we allow 'date' to be optional; if present it should parse)
    required = {"blend_ret", "blend_vol", "blend_sharpe", "cor_penalty"}
    missing = required.difference(df.columns)
    assert not missing, f"missing required columns: {sorted(missing)}"

    # If date is present, ensure it parses
    if "date" in df.columns:
        parsed = pd.to_datetime(df["date"], errors="coerce")
        assert parsed.notna().any(), "date column failed to parse (all NaT)"

    # Ensure metrics are numeric and finite (no NaNs)
    for col in ["blend_ret", "blend_vol", "blend_sharpe", "cor_penalty"]:
        assert pd.api.types.is_numeric_dtype(df[col]), f"{col} must be numeric (got {df[col].dtype})"
        assert df[col].notna().all(), f"{col} has NaNs"

    # A small sanity: at least one row should have non-zero metrics
    non_zero = (
        df["blend_ret"].abs().sum()
        + df["blend_vol"].abs().sum()
        + df["blend_sharpe"].abs().sum()
        + df["cor_penalty"].abs().sum()
    )
    assert non_zero != 0, "all metrics appear to be exactly zero"

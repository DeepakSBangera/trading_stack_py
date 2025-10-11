# tests/test_w6_portfolio_compare.py
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

SCRIPT = Path("scripts/w6_portfolio_compare.py")
OUT = Path("reports/wk6_portfolio_compare.csv")


def _run_report() -> None:
    assert SCRIPT.exists(), f"Expected script not found: {SCRIPT}"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--data-glob",
        "data/csv/*.csv",
        "--out",
        str(OUT),
        "--shrink-alpha",
        "0.2",
    ]
    subprocess.run(cmd, check=True)


def test_w6_portfolio_compare_exists_and_schema() -> None:
    # Generate on fresh CI if missing
    if not OUT.exists():
        _run_report()

    assert OUT.exists(), f"{OUT} not found"

    df = pd.read_csv(OUT)
    assert len(df) > 0, "portfolio compare output is empty"

    required = {"symbol", "weight_ew", "weight_mv_shrink", "adv_cap_ok", "sector_cap_ok"}
    missing = required.difference(df.columns)
    assert not missing, f"missing required columns: {sorted(missing)}"

    # Types & NaNs
    for col in ["weight_ew", "weight_mv_shrink"]:
        assert pd.api.types.is_numeric_dtype(df[col]), f"{col} must be numeric"
        assert df[col].notna().all(), f"{col} has NaNs"
        assert (df[col] >= 0).all(), f"{col} must be non-negative"

    for col in ["adv_cap_ok", "sector_cap_ok"]:
        assert df[col].isin([True, False]).all(), f"{col} must be boolean-like"

    # Weights should roughly sum to 1
    sum_ew = float(df["weight_ew"].sum())
    sum_mv = float(df["weight_mv_shrink"].sum())
    assert 0.95 <= sum_ew <= 1.05, f"weight_ew sums to {sum_ew:.6f}, expected ~1"
    assert 0.95 <= sum_mv <= 1.05, f"weight_mv_shrink sums to {sum_mv:.6f}, expected ~1"

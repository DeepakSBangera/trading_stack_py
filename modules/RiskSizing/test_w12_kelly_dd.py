# tests/test_w12_kelly_dd.py
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

SCRIPT = Path("scripts/w12_kelly_dd.py")
OUT = Path("reports/wk12_kelly_dd.csv")


def _run_report() -> None:
    assert SCRIPT.exists(), f"Missing: {SCRIPT}"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(SCRIPT), "--out", str(OUT)], check=True)


def test_w12_kelly_dd_exists_and_has_core_columns() -> None:
    if not OUT.exists():
        _run_report()
    assert OUT.exists(), f"{OUT} not found"

    df = pd.read_csv(OUT)
    assert len(df) > 0, "empty output"

    required = {"kelly_fraction", "target_vol", "dd_throttle", "position_scale"}
    missing = required.difference(df.columns)
    assert not missing, f"missing columns: {sorted(missing)}"

    for col in required:
        assert pd.api.types.is_numeric_dtype(df[col]), f"{col} must be numeric"
        assert df[col].notna().all(), f"{col} has NaNs"

    if "date" in df.columns:
        parsed = pd.to_datetime(df["date"], errors="coerce")
        assert parsed.notna().any(), "date failed to parse"

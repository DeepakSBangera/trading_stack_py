# tests/test_w11_alpha_blend.py
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

OUT = Path("reports/wk11_alpha_blend.csv")


def _run_report() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "scripts/w11_alpha_blend.py",
            "--out",
            str(OUT),
        ],
        check=True,
    )


def test_w11_alpha_blend_exists_and_has_core_columns() -> None:
    if not OUT.exists():
        _run_report()

    assert OUT.exists(), f"{OUT} not found"

    df = pd.read_csv(OUT)
    assert len(df) >= 1, "no rows in alpha blend report"

    required = {
        "date",
        "blend_ret",
        "blend_vol",
        "blend_sharpe",
        "cor_penalty",
        "source",
        "n_series",
    }
    missing = required.difference(df.columns)
    assert not missing, f"missing columns: {sorted(missing)}"

    # numeric sanity
    for col in ["blend_ret", "blend_vol", "cor_penalty"]:
        assert pd.api.types.is_numeric_dtype(df[col]), f"{col} must be numeric"

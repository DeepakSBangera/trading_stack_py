from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

OUT = Path("reports/wk10_forecast_eval.csv")


def _run_report() -> None:
    """Run the W10 evaluator to (re)create the output CSV."""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "scripts/w10_arimax.py",
        "--data-glob",
        "data/csv/*.csv",
        "--out",
        str(OUT),
        "--order",
        "1,1,1",
        "--test-frac",
        "0.2",
    ]
    subprocess.run(cmd, check=True)


def test_w10_forecast_eval_exists_and_has_core_columns() -> None:
    # 1) Generate the report if missing (fresh CI checkout)
    if not OUT.exists():
        _run_report()

    assert OUT.exists(), f"{OUT} not found"

    # 2) Basic schema checks
    df = pd.read_csv(OUT)
    assert len(df) > 0, "evaluation file is empty"

    required = {"symbol", "model", "rmse", "mae", "mape", "aic", "bic", "fit_status"}
    missing = required.difference(df.columns)
    assert not missing, f"missing required columns: {sorted(missing)}"

    # 3) At least one successful fit
    if "error" in df.columns:
        ok_rows = (df["fit_status"] == "ok") | (df["fit_status"] == "warn_convergence")
        assert ok_rows.any(), "no successful fits found (all rows are errors)"
    else:
        assert (df["fit_status"] == "ok").any() or (
            df["fit_status"] == "warn_convergence"
        ).any(), "fit_status has no ok or warn_convergence rows"

    # 4) Metrics should be numeric and non-negative where applicable
    for col in ["rmse", "mae", "mape"]:
        assert pd.api.types.is_numeric_dtype(df[col]), f"{col} must be numeric"
        assert (df[col] >= 0).all(), f"{col} must be non-negative"

    # AIC/BIC can be negative, but must be numeric
    for col in ["aic", "bic"]:
        assert pd.api.types.is_numeric_dtype(df[col]), f"{col} must be numeric"

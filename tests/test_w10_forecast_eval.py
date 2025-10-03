# tests/test_w10_forecast_eval.py
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

OUT = Path("reports/wk10_forecast_eval.csv")
DATA_GLOB = "data/csv/*.csv"


def _run_report() -> None:
    """
    Run the W10 evaluator to (re)create the output CSV.
    Uses only local files (no network). Fails loudly if the script returns non-zero.
    """
    OUT.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "scripts/w10_arimax.py",
        "--data-glob",
        DATA_GLOB,
        "--out",
        str(OUT),
        "--order",
        "1,1,1",
        "--test-frac",
        "0.2",
    ]

    # Allow CI to optionally provide exogenous regressors via env (keeps test flexible).
    exog_csv = os.environ.get("W10_EXOG_CSV")
    exog_cols = os.environ.get("W10_EXOG_COLS")
    exog_lag = os.environ.get("W10_EXOG_LAG")
    if exog_csv and exog_cols:
        cmd += ["--exog-csv", exog_csv, "--exog-cols", exog_cols]
        if exog_lag:
            cmd += ["--exog-lag", exog_lag]

    # Run and surface stderr/stdout in case of debugging on CI.
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise AssertionError(
            "w10_arimax.py failed.\n"
            f"Command: {' '.join(cmd)}\n\n"
            f"STDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}"
        )


def test_w10_forecast_eval_exists_and_has_core_columns() -> None:
    """
    CI sanity: ensure the W10 evaluator wrote the output CSV and that it
    includes the core metrics columns we expect.
    """
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

    # 3) At least one successful fit (allowing 'warn_convergence' as acceptable)
    assert (
        ((df.get("fit_status") == "ok") | (df.get("fit_status") == "warn_convergence"))
        .fillna(False)
        .any()
    ), "no successful fits found (fit_status has neither 'ok' nor 'warn_convergence')"

    # 4) Metrics should be numeric and non-negative where applicable
    for col in ("rmse", "mae", "mape"):
        assert pd.api.types.is_numeric_dtype(df[col]), f"{col} must be numeric"
        assert (df[col] >= 0).all(), f"{col} must be non-negative"

    # AIC/BIC can be negative, but must be numeric
    for col in ("aic", "bic"):
        assert pd.api.types.is_numeric_dtype(df[col]), f"{col} must be numeric"

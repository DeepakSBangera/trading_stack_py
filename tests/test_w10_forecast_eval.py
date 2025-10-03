# tests/test_w10_forecast_eval.py
from __future__ import annotations

from pathlib import Path

import pandas as pd


def test_w10_forecast_eval_exists_and_has_core_columns() -> None:
    """
    CI sanity: ensure the W10 evaluator wrote the output CSV and that it
    includes the core metrics columns we expect.
    """
    path = Path("reports/wk10_forecast_eval.csv")
    assert path.exists(), "reports/wk10_forecast_eval.csv not found"

    df = pd.read_csv(path)

    core = {"symbol", "model", "rmse", "mae", "mape"}
    missing = core - set(df.columns)
    assert not missing, f"Missing expected columns: {sorted(missing)}"

    # Optional-but-useful fields (do not fail CI if absent)
    # These may appear depending on your script flags:
    # {"aic", "bic", "n_train", "n_test", "exog_used", "fit_status"}

# tests/test_w10_forecast_eval.py
from __future__ import annotations

from pathlib import Path

import pandas as pd


def test_w10_forecast_eval_exists_and_has_core_columns() -> None:
    p = Path("reports/wk10_forecast_eval.csv")
    assert p.exists(), "reports/wk10_forecast_eval.csv not found"

    df = pd.read_csv(p)
    # Must have at least these columns
    required = {"symbol", "model", "rmse", "mae", "mape"}
    missing = required - set(df.columns)
    assert not missing, f"Missing columns: {missing}"

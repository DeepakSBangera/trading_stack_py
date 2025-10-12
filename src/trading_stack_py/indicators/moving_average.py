# src/trading_stack_py/indicators/moving_average.py
from __future__ import annotations

import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    w = int(max(1, window or 1))
    # min_periods = w so the first w-1 are NaN (expected), after that valid
    out = s.rolling(window=w, min_periods=w).mean()
    # if everything somehow NaN (e.g., all non-numeric), fall back to simple ffill
    if out.notna().sum() == 0 and s.notna().any():
        out = s.rolling(window=w, min_periods=1).mean()
    return out

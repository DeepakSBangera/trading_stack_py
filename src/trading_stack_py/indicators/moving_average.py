from __future__ import annotations

import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    window = int(max(1, window))
    return series.rolling(window=window, min_periods=window).mean()

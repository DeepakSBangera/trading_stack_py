from __future__ import annotations

import numpy as np
import pandas as pd


def _as_series(x) -> pd.Series:
    return x if isinstance(x, pd.Series) else pd.Series(x)


def lag(series, k: int = 1):
    s = _as_series(series)
    return s.shift(k)


def rolling_mean(series, window: int = 5):
    s = _as_series(series)
    return s.shift(1).rolling(window=window, min_periods=window).mean()


def rolling_std(series, window: int = 20):
    s = _as_series(series)
    return s.shift(1).rolling(window=window, min_periods=window).std(ddof=1)


def zscore(series, window: int = 20):
    s = _as_series(series)
    mu = s.shift(1).rolling(window=window, min_periods=window).mean()
    sd = s.shift(1).rolling(window=window, min_periods=window).std(ddof=1)
    return (s.shift(1) - mu) / sd


def rsi(series, window: int = 14):
    """RSI using only past data (shifted)."""
    s = _as_series(series)
    r = s.diff()
    gain = r.clip(lower=0).shift(1).rolling(window=window, min_periods=window).mean()
    loss = (-r.clip(upper=0)).shift(1).rolling(window=window, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

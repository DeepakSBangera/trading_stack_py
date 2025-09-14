# Minimal indicators used by the signal rule
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    """Simple moving average."""
    return s.rolling(n, min_periods=n).mean()


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Average True Range (volatility proxy)."""
    high, low_, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low_), (high - prev_close).abs(), (low_ - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume: cumulative volume with sign of price change."""
    direction = np.sign(df["close"].diff()).fillna(0)
    return (direction * df["volume"]).fillna(0).cumsum()


def slope(s: pd.Series, n: int) -> pd.Series:
    """Rolling linear slope over window n."""
    x = np.arange(n)

    def _fit(y: np.ndarray) -> float:
        if len(y) < n or np.isnan(y).any():
            return np.nan
        A = np.vstack([x, np.ones_like(x)]).T
        m, _ = np.linalg.lstsq(A, y.astype(float), rcond=None)[0]
        return float(m)

    return s.rolling(n).apply(lambda w: _fit(w.values), raw=False)


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = s.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

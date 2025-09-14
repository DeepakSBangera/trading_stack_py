# Minimal indicators used by rules (SMA, ATR, OBV, slope, RSI)
import numpy as np
import pandas as pd


def sma(s, n):
    return s.rolling(n, min_periods=n).mean()


def atr(df, n=14):
    high, low_, close = df["high"], df["low"], df["close"]
    prev_c = close.shift(1)
    tr = pd.concat(
        [(high - low_), (high - prev_c).abs(), (low_ - prev_c).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def obv(df):
    # On-Balance Volume: if volume missing (e.g., FX), return zeros
    if "volume" not in df.columns:
        return pd.Series(0.0, index=df.index)
    direction = np.sign(df["close"].diff()).fillna(0)
    vol = df["volume"].fillna(0)
    return (direction * vol).cumsum()


def slope(s, n):
    x = np.arange(n)

    def _fit(y):
        if len(y) < n or np.isnan(y).any():
            return np.nan
        A = np.vstack([x, np.ones_like(x)]).T
        m, _ = np.linalg.lstsq(A, y.astype(float), rcond=None)[0]
        return m

    return s.rolling(n).apply(lambda w: _fit(w.values), raw=False)


def rsi(s, n=14):
    delta = s.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

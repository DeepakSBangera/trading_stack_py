import pandas as pd


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 10) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.rolling(period, min_periods=period).mean()


def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.Series:
    """
    Returns a boolean Series 'in_uptrend' (True for uptrend)
    """
    hl2 = (df["High"] + df["Low"]) / 2.0
    atr_vals = atr(df["High"], df["Low"], df["Close"], period=period)

    upperband = hl2 + multiplier * atr_vals
    lowerband = hl2 - multiplier * atr_vals

    in_uptrend = pd.Series(False, index=df.index)
    final_upper = upperband.copy()
    final_lower = lowerband.copy()

    for i in range(1, len(df)):
        final_upper.iat[i] = (
            min(upperband.iat[i], final_upper.iat[i - 1])
            if df["Close"].iat[i - 1] > final_upper.iat[i - 1]
            else upperband.iat[i]
        )
        final_lower.iat[i] = (
            max(lowerband.iat[i], final_lower.iat[i - 1])
            if df["Close"].iat[i - 1] < final_lower.iat[i - 1]
            else lowerband.iat[i]
        )
        if df["Close"].iat[i] > final_upper.iat[i - 1]:
            in_uptrend.iat[i] = True
        elif df["Close"].iat[i] < final_lower.iat[i - 1]:
            in_uptrend.iat[i] = False
        else:
            in_uptrend.iat[i] = in_uptrend.iat[i - 1]
    return in_uptrend

import pandas as pd


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    delta = close.diff().fillna(0.0)
    direction = delta.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    return (direction * volume.fillna(0)).cumsum()

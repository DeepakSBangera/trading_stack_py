from __future__ import annotations

import pandas as pd

from ..indicators.moving_average import sma
from ..indicators.obv import obv
from ..indicators.supertrend import supertrend


def basic_long_signal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Long-only composite:
    - Supertrend uptrend
    - Price above 50SMA
    - OBV rising (positive slope over 5 days)
    """
    out = df.copy()
    out["SMA50"] = sma(out["Close"], 50)
    out["ST_UP"] = supertrend(out, period=10, multiplier=3.0)
    _obv = obv(out["Close"], out["Volume"]).fillna(0)
    out["OBV_SLOPE5"] = (_obv - _obv.shift(5)).fillna(0)
    out["LONG"] = (out["ST_UP"]) & (out["Close"] > out["SMA50"]) & (out["OBV_SLOPE5"] > 0)
    out["ENTRY"] = out["LONG"] & (~out["LONG"].shift(1).fillna(False))
    out["EXIT"] = (~out["LONG"]) & (out["LONG"].shift(1).fillna(False))
    return out[
        [
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "ENTRY",
            "EXIT",
            "LONG",
            "SMA50",
            "ST_UP",
            "OBV_SLOPE5",
        ]
    ]

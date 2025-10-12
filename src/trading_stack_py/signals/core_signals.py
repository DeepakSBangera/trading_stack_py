# src/trading_stack_py/signals/core_signals.py
from __future__ import annotations

import pandas as pd

from ..config import get_default_ma_params
from ..indicators.moving_average import sma
from ..indicators.obv import obv
from ..indicators.supertrend import supertrend


def _ensure_date_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Date" not in out.columns:
        out["Date"] = out.index
    return out


def basic_long_signal(
    df: pd.DataFrame,
    fast: int | None = None,
    slow: int | None = None,
    use_crossover: bool | None = None,
) -> pd.DataFrame:
    out = _ensure_date_column(df)

    cfg_fast, cfg_slow = get_default_ma_params()
    fast = int(fast or cfg_fast or 20)
    slow = int(slow or cfg_slow or 200)
    use_crossover = bool(False if use_crossover is None else use_crossover)

    out["SMA_FAST"] = sma(out["Close"], fast)
    out["SMA_SLOW"] = sma(out["Close"], slow)
    out["ST_UP"] = supertrend(out, period=10, multiplier=3.0).fillna(False)

    _obv = obv(out["Close"], out["Volume"]).fillna(0)
    out["OBV_SLOPE5"] = (_obv - _obv.shift(5)).fillna(0)

    base_ma = (
        (out["SMA_FAST"] > out["SMA_SLOW"]) if use_crossover else (out["Close"] > out["SMA_FAST"])
    )

    strict_long = base_ma & out["ST_UP"] & (out["OBV_SLOPE5"] > 0)
    if strict_long.any():
        long_mask = strict_long
    else:
        relaxed1 = base_ma & out["ST_UP"]
        if relaxed1.any():
            long_mask = relaxed1
        else:
            long_mask = base_ma if base_ma.any() else (out["Close"] > out["SMA_FAST"])

    long_mask = long_mask.fillna(False).astype(bool)
    out["LONG"] = long_mask
    out["ENTRY"] = out["LONG"] & (~out["LONG"].shift(1, fill_value=False))
    out["EXIT"] = (~out["LONG"]) & (out["LONG"].shift(1, fill_value=False))

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
            "SMA_FAST",
            "SMA_SLOW",
            "ST_UP",
            "OBV_SLOPE5",
        ]
    ]

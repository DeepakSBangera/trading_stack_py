# src/trading_stack_py/signals/core_signals.py
from __future__ import annotations

import pandas as pd

from ..config import StrategyConfig
from ..indicators.moving_average import sma
from ..indicators.obv import obv
from ..indicators.supertrend import supertrend


def _coerce_series(x: pd.Series) -> pd.Series:
    # Ensure numeric dtype; leave NaNs for indicators to handle
    return pd.to_numeric(x, errors="coerce")


def basic_long_signal(
    df: pd.DataFrame,
    *,
    use_crossover: bool | None = None,  # None => use cfg.use_crossover; else override
    cfg: StrategyConfig | None = None,
    fast: int | None = None,  # None => use cfg.ma_fast; else override
    slow: int | None = None,  # None => use cfg.ma_slow; else override
) -> pd.DataFrame:
    """
    Minimal, robust long-only signal scaffold.

    Logic:
      - If use_crossover is True:  LONG = SMA(fast) > SMA(slow)
      - Else (default):            LONG = Close > SMA(fast)
    ATR/Supertrend and OBV are computed but *not required* for entry,
    so you won’t accidentally filter out every trade on synthetic data.

    ENTRY = LONG and not LONG.shift(1)
    EXIT  = not LONG and LONG.shift(1)
    """
    out = df.copy()

    # Normalise columns we rely on
    for col in ("Date", "Open", "High", "Low", "Close", "Volume"):
        if col not in out.columns:
            raise ValueError(f"basic_long_signal: missing required column '{col}'")
    out["Close"] = _coerce_series(out["Close"])
    out["Open"] = _coerce_series(out["Open"])
    out["High"] = _coerce_series(out["High"])
    out["Low"] = _coerce_series(out["Low"])
    out["Volume"] = _coerce_series(out["Volume"])

    # Resolve parameters with sane fallbacks
    cfg_fast = getattr(cfg, "ma_fast", None)
    cfg_slow = getattr(cfg, "ma_slow", None)
    cfg_use_x = getattr(cfg, "use_crossover", None)

    fast_n = int(fast or cfg_fast or 20)
    slow_n = int(slow or cfg_slow or 200)
    use_x = bool(
        use_crossover
        if use_crossover is not None
        else (cfg_use_x if cfg_use_x is not None else False)
    )

    # Moving averages
    close = out["Close"]
    out["SMA_FAST"] = sma(close, fast_n)
    out["SMA_SLOW"] = sma(close, slow_n)

    # Lightweight context fields (not gating entries)
    try:
        out["ST_UP"] = supertrend(out, period=10, multiplier=3.0)
    except Exception:
        out["ST_UP"] = False
    try:
        _obv = obv(out["Close"], out["Volume"]).fillna(0.0)
        out["OBV_SLOPE5"] = (_obv - _obv.shift(5)).fillna(0.0)
    except Exception:
        out["OBV_SLOPE5"] = 0.0

    # Core rule
    if use_x:
        long_mask = out["SMA_FAST"] > out["SMA_SLOW"]
    else:
        long_mask = out["Close"] > out["SMA_FAST"]

    # Ensure we don’t generate signals before MAs exist
    long_mask &= out["SMA_FAST"].notna()
    if use_x:
        long_mask &= out["SMA_SLOW"].notna()

    out["LONG"] = long_mask
    out["ENTRY"] = out["LONG"] & (~out["LONG"].shift(1, fill_value=False))
    out["EXIT"] = (~out["LONG"]) & (out["LONG"].shift(1, fill_value=False))

    # Ordered, compact output
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

from __future__ import annotations

import pandas as pd

from ..indicators.moving_average import sma


def basic_long_signal(
    df: pd.DataFrame,
    *,
    use_crossover: bool = False,
    fast: int = 20,
    slow: int = 100,
) -> pd.DataFrame:
    """
    Minimal MA signal generator (no Supertrend/OBV).

    Modes
    -----
    - use_crossover = False  -> LONG when Close > SMA(fast)
    - use_crossover = True   -> LONG when SMA(fast) > SMA(slow)

    Notes
    -----
    - Handles warm-up by filling NaN comparisons with False.
    - Returns the original OHLCV plus ENTRY/EXIT/LONG/SMA_* columns.
    """
    # Work on a copy and ensure Date is a column for downstream CSV
    out = df.copy()
    if "Date" not in out.columns:
        out = out.reset_index().rename(columns={"index": "Date"})

    close = out["Close"]

    # Compute SMAs (robust to short series)
    out["SMA_FAST"] = sma(close, fast)
    out["SMA_SLOW"] = sma(close, slow)

    if use_crossover:
        cond = out["SMA_FAST"] > out["SMA_SLOW"]
    else:
        cond = close > out["SMA_FAST"]

    # Fill NaNs from warm-up with False
    out["LONG"] = cond.fillna(False)

    # ENTRY = first True after a False; EXIT = first False after a True
    prev = out["LONG"].shift(1, fill_value=False)
    out["ENTRY"] = out["LONG"] & (~prev)
    out["EXIT"] = (~out["LONG"]) & prev

    # Return tidy columns
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
        ]
    ]

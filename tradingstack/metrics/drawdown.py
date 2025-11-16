from __future__ import annotations

import numpy as np
import pandas as pd


def _as_returns(r: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(r, pd.DataFrame):
        # prefer a column named 'return' else first numeric col
        for c in r.columns:
            s = pd.to_numeric(r[c], errors="coerce")
            if s.notna().any():
                return pd.Series(s.values, index=r.index)
        raise ValueError("Could not infer return series from DataFrame.")
    return pd.to_numeric(r, errors="coerce")


def max_drawdown(returns: pd.Series | pd.DataFrame) -> float:
    """
    Max drawdown magnitude (positive number). Input is arithmetic returns.
    """
    r = _as_returns(returns).fillna(0.0)
    equity = (1.0 + r).cumprod()
    rolling_peak = equity.cummax()
    dd = (equity / rolling_peak) - 1.0  # negative or zero
    return float(np.nanmin(dd)) * -1.0  # return positive magnitude

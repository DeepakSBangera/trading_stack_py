# tradingstack/metrics/omega.py
import numpy as np
import pandas as pd


def omega_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
    """
    Omega ratio at threshold (default 0.0).
    Omega = E[(r - t)+] / E[(t - r)+], computed on per-period returns.
    """
    r = (
        pd.to_numeric(pd.Series(returns), errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    gains = (r - threshold).clip(lower=0.0)
    losses = (threshold - r).clip(lower=0.0)
    denom = losses.mean()
    if denom <= 1e-12:
        # all mass at/above threshold
        return float(np.inf)
    return float(gains.mean() / denom)

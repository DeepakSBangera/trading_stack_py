"""
Sortino ratio (annualized).
- sortino_annual(rets: pd.Series, rf: float = 0.0, ann_factor: int = 252) -> float
"""

import numpy as np
import pandas as pd


def _as_returns_series(rets_like) -> pd.Series:
    s = (
        pd.Series(rets_like)
        if not isinstance(rets_like, pd.Series)
        else rets_like.copy()
    )
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return s.astype(float)


def sortino_annual(rets_like, rf: float = 0.0, ann_factor: int = 252) -> float:
    """
    Annualized Sortino: mean(excess) / std(negatives) * sqrt(ann_factor)
    where 'negatives' are returns < rf on the same frequency as rets_like.
    """
    r = _as_returns_series(rets_like)
    excess = r - rf
    downside = excess[excess < 0.0]
    if downside.empty:
        # no downside risk → formally infinite; return large sentinel
        return float("inf")
    downside_std = float(downside.std(ddof=1)) or 0.0
    if downside_std == 0.0:
        return float("inf")
    mean_excess = float(excess.mean())
    return (mean_excess / downside_std) * np.sqrt(float(ann_factor))

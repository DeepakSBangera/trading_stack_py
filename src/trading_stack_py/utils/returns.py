from __future__ import annotations

import numpy as np


def to_excess_returns(returns: np.ndarray, rf: float = 0.0, freq: str = "D") -> np.ndarray:
    r = np.asarray(returns, dtype=float)
    if rf != 0:
        f = freq.upper()
        if f in ("D", "B"):
            period_rf = rf / 252.0
        elif f in ("W",):
            period_rf = rf / 52.0
        elif f in ("M",):
            period_rf = rf / 12.0
        elif f in ("Q",):
            period_rf = rf / 4.0
        else:
            period_rf = rf
        r = r - period_rf
    return r


def sharpe_ratio(returns: np.ndarray, rf: float = 0.0, freq: str = "D", ddof: int = 1) -> float:
    r = to_excess_returns(returns, rf, freq)
    mu = r.mean()
    sd = r.std(ddof=ddof)
    return 0.0 if sd == 0 else float(mu / sd)

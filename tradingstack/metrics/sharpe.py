from __future__ import annotations

import numpy as np
import pandas as pd


def _to_series(x) -> pd.Series:
    if isinstance(x, pd.Series):
        return x
    if isinstance(x, (list, tuple)):
        return pd.Series(x, dtype=float)
    return pd.Series(pd.to_numeric(x, errors="coerce"), dtype=float)


def sharpe_daily(rets, risk_free_daily: float = 0.0, eps: float = 1e-12) -> float:
    """
    Daily Sharpe given simple daily returns (not log returns).
    """
    r = _to_series(rets).astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # subtract daily risk-free if provided
    r = r - float(risk_free_daily)
    mu = r.mean()
    sd = r.std(ddof=1)
    return float(mu / (sd + eps))


def sharpe_annual(
    rets, ann_factor: float = 252.0, risk_free_annual: float = 0.0, eps: float = 1e-12
) -> float:
    """
    Annualized Sharpe from daily simple returns.
    `ann_factor` is the trading days per year.
    """
    r = _to_series(rets).astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # convert annual RF to daily, subtract
    rf_daily = (
        (1.0 + float(risk_free_annual)) ** (1.0 / ann_factor) - 1.0 if risk_free_annual else 0.0
    )
    r = r - rf_daily
    mu = r.mean()
    sd = r.std(ddof=1)
    return float((mu / (sd + eps)) * np.sqrt(ann_factor))

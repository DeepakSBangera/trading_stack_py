# src/trading_stack_py/metrics/performance.py
from __future__ import annotations

from math import sqrt

import numpy as np
import pandas as pd
from scipy.stats import norm


def max_drawdown(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    dd = (equity / roll_max) - 1.0
    return dd.min()


def sharpe_daily(returns: pd.Series, rf_daily: float = 0.0) -> float:
    r = returns - rf_daily
    if r.std(ddof=0) == 0:
        return 0.0
    return float((np.sqrt(252) * r.mean()) / r.std(ddof=0))


def cagr(equity: pd.Series, periods_per_year: int = 252) -> float:
    if len(equity) < 2:
        return 0.0
    total_return = float(equity.iloc[-1] / equity.iloc[0])
    years = len(equity) / periods_per_year
    if years <= 0:
        return 0.0
    return total_return ** (1 / years) - 1.0


def calmar(equity: pd.Series) -> float:
    dd = max_drawdown(equity)
    if dd == 0:
        return 0.0
    return cagr(equity) / abs(dd)


def summarize(bt_df: pd.DataFrame) -> dict:
    eq = bt_df["Equity"]
    ret = bt_df["Return"]
    return {
        "CAGR": cagr(eq),
        "Sharpe": sharpe_daily(ret),
        "MaxDD": max_drawdown(eq),
        "Calmar": calmar(eq),
        "Trades": int(bt_df["ENTRY"].sum() + bt_df["EXIT"].sum()),
    }


def probabilistic_sharpe_ratio(
    sr: float, sr_bench: float, n: int, skew: float = 0.0, kurt: float = 3.0
) -> float:
    """
    PSR from Bailey & López de Prado (2012/2014).
    sr: observed Sharpe (daily)
    sr_bench: benchmark Sharpe (e.g., 0)
    n: number of returns used to compute sr
    skew, kurt: skewness and kurtosis of returns (excess kurt not required here; use full kurtosis)
    """
    if n <= 2:
        return 0.0
    num = (sr - sr_bench) * sqrt(n - 1)
    den = sqrt(1 - skew * sr + (kurt - 1) / 4.0 * (sr**2))
    z = num / den if den > 0 else 0.0
    return float(norm.cdf(z))

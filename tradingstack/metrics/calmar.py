# tradingstack/metrics/calmar.py
import numpy as np
import pandas as pd

from .drawdown import max_drawdown


def _as_clean_nav(nav: pd.Series) -> pd.Series:
    s = pd.to_numeric(pd.Series(nav), errors="coerce").replace([np.inf, -np.inf], np.nan)
    s = s.ffill().fillna(1.0)
    return s


def _cagr_from_nav(nav: pd.Series, *, periods_per_year: int = 252) -> float:
    s = _as_clean_nav(nav)
    n = len(s)
    if n < 2:
        return 0.0
    start = float(s.iloc[0])
    end = float(s.iloc[-1])
    if start <= 0:
        return 0.0
    total_ret = end / start
    years = n / periods_per_year
    if years <= 0:
        return 0.0
    return float(total_ret ** (1.0 / years) - 1.0)


def calmar_ratio(nav: pd.Series, *, periods_per_year: int = 252) -> float:
    """
    Calmar = CAGR / MaxDrawdown. Returns 0.0 if MDD is ~0 to avoid blow-ups.
    """
    cagr = _cagr_from_nav(nav, periods_per_year=periods_per_year)
    mdd = max_drawdown(nav)
    if mdd <= 1e-12:
        return 0.0
    return float(cagr / mdd)

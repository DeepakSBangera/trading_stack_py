# Performance metrics used for robustness gates
import numpy as np
import pandas as pd

def to_series(x):
    return x if isinstance(x, pd.Series) else pd.Series(x)

def dd_curve(equity: pd.Series) -> pd.Series:
    equity = to_series(equity).astype(float)
    peak = equity.cummax()
    return equity / peak - 1.0

def max_drawdown(equity: pd.Series) -> float:
    return float(dd_curve(equity).min())

def cagr(equity: pd.Series, periods_per_year: int = 252) -> float:
    eq = to_series(equity).astype(float)
    if len(eq) < 2 or eq.iloc[0] <= 0:
        return np.nan
    years = len(eq) / periods_per_year
    return float((eq.iloc[-1] / eq.iloc[0]) ** (1 / max(years, 1e-9)) - 1)

def sharpe(returns: pd.Series, periods_per_year: int = 252, rf: float = 0.0) -> float:
    r = to_series(returns).astype(float)
    if r.std(ddof=0) == 0:
        return np.nan
    mu = r.mean() - rf / periods_per_year
    sigma = r.std(ddof=0)
    return float((mu / sigma) * np.sqrt(periods_per_year))

def sortino(returns: pd.Series, periods_per_year: int = 252, rf: float = 0.0) -> float:
    r = to_series(returns).astype(float)
    dn = r[r < 0]
    if dn.std(ddof=0) == 0:
        return np.nan
    mu = r.mean() - rf / periods_per_year
    sigma_dn = dn.std(ddof=0)
    return float((mu / sigma_dn) * np.sqrt(periods_per_year))

def profit_factor(returns: pd.Series) -> float:
    r = to_series(returns).astype(float)
    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()
    if losses <= 0:
        return np.inf
    return float(gains / losses)

def calmar(cagr_val: float, mdd: float) -> float:
    if mdd >= 0:
        return np.inf
    return float(cagr_val / abs(mdd))

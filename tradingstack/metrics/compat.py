from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from importlib import import_module

import numpy as np
import pandas as pd


def _resolve(mod: str, candidates: Iterable[str]) -> Callable | None:
    """Return the first callable found among candidate names in module, else None."""
    try:
        m = import_module(mod)
    except Exception:
        return None
    for name in candidates:
        fn = getattr(m, name, None)
        if callable(fn):
            return fn
    return None


def _ann(index) -> int:
    # default daily
    return 252


# -------- fallbacks (used only if your native functions are absent) ----------
def _fb_sharpe_ratio(
    returns: pd.Series, rf: float = 0.0, annualization: int | None = None
) -> float:
    r = pd.to_numeric(returns, errors="coerce").astype("float64").dropna()
    if r.empty:
        return np.nan
    if annualization is None:
        annualization = _ann(r.index)
    excess = r - rf
    sd = excess.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return np.nan
    return (excess.mean() / sd) * math.sqrt(annualization)


def _fb_sortino_ratio(
    returns: pd.Series, rf: float = 0.0, annualization: int | None = None
) -> float:
    r = pd.to_numeric(returns, errors="coerce").astype("float64").dropna()
    if r.empty:
        return np.nan
    if annualization is None:
        annualization = _ann(r.index)
    excess = r - rf
    downside = excess.clip(upper=0.0)
    dd = math.sqrt((downside.pow(2)).mean())
    if dd == 0 or np.isnan(dd):
        return np.nan
    return (excess.mean() / dd) * math.sqrt(annualization)


def _fb_max_drawdown(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna().astype("float64")
    if s.empty:
        return np.nan
    nav = s if (s.min() > 0 and s.median() > 0.5) else (1.0 + s).cumprod()
    peak = nav.cummax()
    dd = nav / peak - 1.0
    return float(dd.min())


def _fb_calmar_ratio(series: pd.Series, annualization: int | None = None) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna().astype("float64")
    if s.empty:
        return np.nan
    nav = s if (s.min() > 0 and s.median() > 0.5) else (1.0 + s).cumprod()
    if annualization is None:
        annualization = _ann(nav.index)
    n = len(nav)
    if n < 2:
        return np.nan
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (annualization / n) - 1.0
    mdd = abs(_fb_max_drawdown(nav))
    if mdd == 0 or np.isnan(mdd):
        return np.nan
    return cagr / mdd


def _fb_omega_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
    r = pd.to_numeric(returns, errors="coerce").astype("float64").dropna()
    if r.empty:
        return np.nan
    gains = (r - threshold).clip(lower=0.0).sum()
    losses = (threshold - r).clip(lower=0.0).sum()
    if losses == 0:
        return np.inf
    return float(gains / losses)


# -------- bind to your existing modules if present ---------------------------
_sharpe_impl = _resolve("tradingstack.metrics.sharpe", ["sharpe_ratio", "sharpe", "calc_sharpe"])
_sortino_impl = _resolve(
    "tradingstack.metrics.sortino", ["sortino_ratio", "sortino", "calc_sortino"]
)
_mdd_impl = _resolve(
    "tradingstack.metrics.drawdown", ["max_drawdown", "drawdown", "calc_max_drawdown"]
)
_calmar_impl = _resolve("tradingstack.metrics.calmar", ["calmar_ratio", "calmar", "calc_calmar"])
_omega_impl = _resolve("tradingstack.metrics.omega", ["omega_ratio", "omega", "calc_omega"])


def sharpe_ratio(returns: pd.Series, rf: float = 0.0, annualization: int | None = None) -> float:
    if _sharpe_impl:  # type: ignore[truthy-function]
        return _sharpe_impl(returns, rf=rf, annualization=annualization or _ann(returns.index))
    return _fb_sharpe_ratio(returns, rf=rf, annualization=annualization)


def sortino_ratio(returns: pd.Series, rf: float = 0.0, annualization: int | None = None) -> float:
    if _sortino_impl:
        return _sortino_impl(returns, rf=rf, annualization=annualization or _ann(returns.index))
    return _fb_sortino_ratio(returns, rf=rf, annualization=annualization)


def max_drawdown(series: pd.Series) -> float:
    if _mdd_impl:
        return _mdd_impl(series)
    return _fb_max_drawdown(series)


def calmar_ratio(series: pd.Series, annualization: int | None = None) -> float:
    if _calmar_impl:
        return _calmar_impl(series, annualization=annualization or _ann(series.index))
    return _fb_calmar_ratio(series, annualization=annualization)


def omega_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
    if _omega_impl:
        return _omega_impl(returns, threshold=threshold)
    return _fb_omega_ratio(returns, threshold=threshold)


__all__ = [
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "calmar_ratio",
    "omega_ratio",
]

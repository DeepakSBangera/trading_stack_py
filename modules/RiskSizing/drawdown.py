# tradingstack/metrics/drawdown.py
import numpy as np
import pandas as pd


def _as_clean_series(x: pd.Series, *, kind: str) -> pd.Series:
    """
    kind='nav'  -> forward-fill then final fill with 1.0 (sensible for NAV/price streams)
    kind='rets' -> forward-fill then final fill with 0.0 (sensible for returns)
    """
    s = pd.to_numeric(pd.Series(x), errors="coerce").replace([np.inf, -np.inf], np.nan)
    s = s.ffill()  # pandas >= 1.5 style; avoids deprecated fillna(method="ffill")
    if kind == "nav":
        s = s.fillna(1.0)
    else:
        s = s.fillna(0.0)
    return s


def max_drawdown(nav: pd.Series) -> float:
    """
    Max drawdown measured on a NAV/price series in [0, 1].
    """
    s = _as_clean_series(nav, kind="nav")
    if len(s) == 0:
        return 0.0
    run_max = s.cummax()
    dd = (run_max - s) / run_max.replace(0, np.nan)
    dd = dd.fillna(0.0)
    return float(dd.max())

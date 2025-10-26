from __future__ import annotations

import numpy as np
import pandas as pd

# Optional parity references to point metrics come from the compat shim (safe & stable).
# They are not required for the rolling computations below, but are here for parity/testing use.
try:
    from tradingstack.metrics.compat import (
        max_drawdown as _point_max_dd,  # noqa: F401
    )
    from tradingstack.metrics.compat import (
        sharpe_ratio as _point_sharpe,  # noqa: F401
    )
    from tradingstack.metrics.compat import (
        sortino_ratio as _point_sortino,  # noqa: F401
    )
except Exception:
    # Parity functions are optional; rolling computations do not depend on them.
    _point_sharpe = _point_sortino = _point_max_dd = None  # type: ignore

# Date normalization utility (ensure tz-naive, sorted DatetimeIndex named 'date')
try:
    from tradingstack.utils.dates import normalize_date_index
except Exception:
    # Minimal inline fallback to keep the module self-sufficient
    def normalize_date_index(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:  # type: ignore
        if date_col in df.columns:
            dt = pd.to_datetime(df[date_col], utc=True, errors="coerce")
            # strip tz -> tz-naive
            try:
                dt = dt.dt.tz_convert(None)
            except Exception:
                pass
            try:
                dt = dt.dt.tz_localize(None)
            except Exception:
                pass
            df = df.assign(**{date_col: dt}).dropna(subset=[date_col]).sort_values(date_col)
            df = df.drop_duplicates(subset=[date_col]).set_index(date_col)
        else:
            idx = pd.to_datetime(df.index, utc=True, errors="coerce")
            try:
                idx = idx.tz_convert(None)
            except Exception:
                pass
            try:
                idx = idx.tz_localize(None)
            except Exception:
                pass
            df.index = idx
            df = df.dropna(axis=0, how="any").sort_index()
        df.index.name = "date"
        return df


# ---------- helpers -----------------------------------------------------------


def _to_ret_from_nav(nav: pd.Series) -> pd.Series:
    nav = pd.to_numeric(nav, errors="coerce").astype("float64")
    ret = nav.pct_change()
    return ret.replace([np.inf, -np.inf], np.nan).dropna()


# ---------- rolling metrics ---------------------------------------------------


def rolling_volatility(returns: pd.Series, window: int = 63, annualization: int = 252) -> pd.Series:
    """
    Annualized rolling volatility over 'window' periods.
    """
    r = pd.to_numeric(returns, errors="coerce").astype("float64")
    vol = r.rolling(window=window, min_periods=window).std(ddof=1)
    return vol * np.sqrt(annualization)


def rolling_sharpe(
    returns: pd.Series, window: int = 252, annualization: int = 252, rf: float = 0.0
) -> pd.Series:
    """
    Rolling Sharpe ratio using mean/std of excess returns in each window.
    """
    r = pd.to_numeric(returns, errors="coerce").astype("float64")
    excess = r - rf
    mean = excess.rolling(window=window, min_periods=window).mean()
    std = excess.rolling(window=window, min_periods=window).std(ddof=1)
    sharpe = (mean / std) * np.sqrt(annualization)
    return sharpe.replace([np.inf, -np.inf], np.nan)


def rolling_sortino(
    returns: pd.Series, window: int = 252, annualization: int = 252, rf: float = 0.0
) -> pd.Series:
    """
    Rolling Sortino ratio using downside deviation within each window.
    """
    r = pd.to_numeric(returns, errors="coerce").astype("float64")
    excess = r - rf
    downside = excess.clip(upper=0.0)
    dd = np.sqrt((downside.pow(2)).rolling(window=window, min_periods=window).mean())
    mu = excess.rolling(window=window, min_periods=window).mean()
    sortino = (mu / dd) * np.sqrt(annualization)
    return sortino.replace([np.inf, -np.inf], np.nan)


def rolling_drawdown(nav: pd.Series, window: int = 252) -> pd.Series:
    """
    Rolling maximum drawdown (<= 0) computed over a sliding window on NAV.
    """
    x = pd.to_numeric(nav, errors="coerce").astype("float64")

    def _window_mdd(s: pd.Series) -> float:
        peak = s.cummax()
        dd = s / peak - 1.0
        return float(dd.min())

    return x.rolling(window=window, min_periods=window).apply(_window_mdd, raw=False)


def trend_regime(
    nav: pd.Series, fast: int = 50, slow: int = 200, method: str = "sma_crossover"
) -> pd.Series:
    """
    - 'sma_crossover': +1 if SMA(fast) > SMA(slow), -1 if <, 0 otherwise.
    - 'zscore': sign(zscore(nav, slow)).
    Returns int8 series in {-1, 0, +1}.
    """
    x = pd.to_numeric(nav, errors="coerce").astype("float64")

    if method == "sma_crossover":
        f = x.rolling(fast, min_periods=fast).mean()
        s = x.rolling(slow, min_periods=slow).mean()
        reg = pd.Series(0, index=x.index, dtype="int8")
        reg = reg.mask(f > s, 1)
        reg = reg.mask(f < s, -1)
        return reg

    if method == "zscore":
        m = x.rolling(slow, min_periods=slow).mean()
        sd = x.rolling(slow, min_periods=slow).std(ddof=1)
        z = (x - m) / sd
        reg = pd.Series(0, index=x.index, dtype="int8")
        reg = reg.mask(z > 0, 1)
        reg = reg.mask(z < 0, -1)
        return reg

    raise ValueError(f"Unknown trend regime method: {method}")


def compute_rolling_metrics_from_nav(
    df: pd.DataFrame,
    nav_col: str = "_nav",
    ret_window_sharpe: int = 252,
    ret_window_sortino: int = 252,
    vol_window: int = 63,
    dd_window: int = 252,
    annualization: int = 252,
    rf_per_period: float = 0.0,
    regime_method: str = "sma_crossover",
    regime_fast: int = 50,
    regime_slow: int = 200,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    df : DataFrame
        Contains either a 'date' column or a DatetimeIndex, plus a NAV column.
    nav_col : str
        Column in df containing NAV (strictly positive).
    Other params:
        Window sizes, annualization, risk-free per period, and regime settings.

    Returns
    -------
    DataFrame indexed by 'date' with columns:
        - rolling_vol
        - rolling_sharpe
        - rolling_sortino
        - rolling_mdd
        - regime  (int8 in {-1, 0, +1})
    """
    df = normalize_date_index(df, "date")
    if nav_col not in df.columns:
        raise KeyError(f"Missing NAV column '{nav_col}' in DataFrame columns: {list(df.columns)}")

    nav = pd.to_numeric(df[nav_col], errors="coerce").astype("float64")
    rets = _to_ret_from_nav(nav)

    out = pd.DataFrame(index=df.index)
    out["rolling_vol"] = rolling_volatility(rets, window=vol_window, annualization=annualization)
    out["rolling_sharpe"] = rolling_sharpe(
        rets, window=ret_window_sharpe, annualization=annualization, rf=rf_per_period
    )
    out["rolling_sortino"] = rolling_sortino(
        rets, window=ret_window_sortino, annualization=annualization, rf=rf_per_period
    )
    out["rolling_mdd"] = rolling_drawdown(nav, window=dd_window)
    out["regime"] = trend_regime(
        nav, fast=regime_fast, slow=regime_slow, method=regime_method
    ).astype("int8")
    return out

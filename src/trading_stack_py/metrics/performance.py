from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def _safe_returns(df: pd.DataFrame) -> pd.Series:
    """
    Return a daily returns series with index aligned to df:
    - Prefer 'Return' if present
    - Else compute from 'Equity' if present
    - Else compute from 'Close'
    """
    if "Return" in df.columns:
        ret = pd.Series(df["Return"]).astype(float)
        return ret.fillna(0.0)

    if "Equity" in df.columns:
        eq = pd.Series(df["Equity"]).astype(float)
        ret = eq.pct_change()
        return ret.fillna(0.0)

    if "Close" in df.columns:
        px = pd.Series(df["Close"]).astype(float)
        ret = px.pct_change()
        return ret.fillna(0.0)

    raise KeyError("No suitable columns to compute returns (need 'Return' or 'Equity' or 'Close').")


def _safe_equity(df: pd.DataFrame, base: float = 1.0) -> pd.Series:
    """
    Return an equity curve:
    - Prefer 'Equity' if present
    - Else cumprod of (1+Return) or computed returns from Close
    """
    if "Equity" in df.columns:
        eq = pd.Series(df["Equity"]).astype(float)
        # guard against zeros/negatives if any
        return eq.replace([np.inf, -np.inf], np.nan).fillna(method="ffill").fillna(method="bfill")

    ret = _safe_returns(df)
    eq = pd.Series(base * (1.0 + ret).cumprod(), index=df.index)
    return eq


def _max_drawdown(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    return float(dd.min()) if len(dd) else 0.0


def _sharpe(ret: pd.Series, periods_per_year: int = 252) -> float:
    if ret.std(ddof=0) == 0 or len(ret) == 0:
        return 0.0
    return float((ret.mean() / ret.std(ddof=0)) * np.sqrt(periods_per_year))


def _cagr(equity: pd.Series, periods_per_year: int = 252) -> float:
    if len(equity) == 0:
        return 0.0
    start = float(equity.iloc[0])
    end = float(equity.iloc[-1])
    # trading days approximation:
    years = max(1e-9, len(equity) / periods_per_year)
    if start <= 0 or end <= 0:
        return 0.0
    return float((end / start) ** (1.0 / years) - 1.0)


def _infer_trades(df: pd.DataFrame) -> int | None:
    """
    Try to infer number of trades:
    - Prefer last row 'Trades' column if present
    - Else sum of ENTRY booleans if present
    - Else count rising edges of Position > 0 if present
    - Else None
    """
    if "Trades" in df.columns and len(df) > 0:
        try:
            return int(df["Trades"].iloc[-1])
        except Exception:
            pass

    if "ENTRY" in df.columns:
        try:
            return int(pd.Series(df["ENTRY"]).fillna(False).astype(bool).sum())
        except Exception:
            pass

    if "Position" in df.columns:
        pos = pd.Series(df["Position"]).fillna(0)
        edges = ((pos > 0) & (pos.shift(1).fillna(0) <= 0)).sum()
        try:
            return int(edges)
        except Exception:
            pass

    return None


def summarize(bt_df: pd.DataFrame) -> dict[str, float]:
    """
    Compute summary stats from a backtest dataframe.
    Accepts either a full panel with 'Equity' or legacy frames with 'Return'.
    """
    ret = _safe_returns(bt_df)
    eq = _safe_equity(bt_df)

    sharpe = _sharpe(ret)
    mdd = _max_drawdown(eq)
    cagr = _cagr(eq)
    calmar = float(cagr / abs(mdd)) if mdd < 0 else 0.0

    trades = _infer_trades(bt_df)
    out = {
        "CAGR": cagr,
        "Sharpe": sharpe,
        "MaxDD": mdd,
        "Calmar": calmar,
    }
    if trades is not None:
        out["Trades"] = float(trades)
    return out


# ---- Optional: Probabilistic Sharpe Ratio helpers if you need them elsewhere ----
def probabilistic_sharpe_ratio(observed_sr: float, benchmark_sr: float, n_obs: int) -> float:
    """
    Bailey & Lopez de Prado PSR (simplified).
    observed_sr: observed Sharpe ratio
    benchmark_sr: Sharpe threshold to test against
    n_obs: number of observations
    """
    if n_obs <= 1:
        return 0.0
    z = (observed_sr - benchmark_sr) * np.sqrt(n_obs - 1)
    return float(1.0 - norm.cdf(z))

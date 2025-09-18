from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    curve: pd.DataFrame  # columns: ret, equity
    weights: pd.DataFrame  # daily weights per ticker
    stats: dict[str, float]  # summary metrics


def momentum_12_1(px: pd.DataFrame, lb12: int = 252, lb1: int = 21) -> pd.DataFrame:
    """
    12-1 momentum: return over last 12m excluding last 1m.
    r12_1(t) = (1+r_252) / (1+r_21) - 1
    Implemented with prices:
      P_{t-21} / P_{t-252-21} - 1  vs  P_t / P_{t-21} - 1
    """
    px = px.sort_index()
    p_t_minus_21 = px.shift(lb1)
    r_252 = p_t_minus_21 / px.shift(lb12 + lb1) - 1.0
    r_21 = px / p_t_minus_21 - 1.0
    r_12_1 = (1.0 + r_252) / (1.0 + r_21) - 1.0
    return r_12_1


def run_backtest(
    px: pd.DataFrame,
    top_n: int = 5,
    rebalance: str = "W-FRI",
    tc_bps: float = 5.0,
    start: str | None = None,
    capital: float = 100_000.0,
) -> BacktestResult:
    """
    Very small vectorized backtest:
    - rank tickers by 12-1 momentum each rebalance date
    - hold top-N equal-weight until next rebalance
    - apply simple transaction costs on rebalance days
    """
    px = px.sort_index()
    if start:
        px = px.loc[pd.to_datetime(start) :].copy()

    # Daily simple returns
    daily_ret = px.pct_change().fillna(0.0)

    # Scores and rebalance dates
    scores = momentum_12_1(px)
    rebal_dates = scores.resample(rebalance).last().dropna(how="all").index

    # Containers
    weights = pd.DataFrame(0.0, index=daily_ret.index, columns=daily_ret.columns)
    turnover = pd.Series(0.0, index=daily_ret.index, dtype=float)
    prev_w = pd.Series(0.0, index=daily_ret.columns)

    # Build weights piecewise by rebalance window
    for i, dt in enumerate(rebal_dates):
        s = scores.loc[dt].dropna().sort_values(ascending=False)
        picks = s.head(top_n).index.tolist()

        w_target = pd.Series(0.0, index=daily_ret.columns)
        if picks:
            w_target[picks] = 1.0 / len(picks)

        next_dt = (
            rebal_dates[i + 1]
            if i + 1 < len(rebal_dates)
            else weights.index.max() + pd.Timedelta(days=1)
        )
        mask = (weights.index >= dt) & (weights.index < next_dt)
        weights.loc[mask] = w_target.values

        # Transaction cost charged on the rebalance day
        turnover.loc[dt] = (w_target - prev_w).abs().sum()
        prev_w = w_target

    tc = float(tc_bps) / 10_000.0
    port_ret = (weights * daily_ret).sum(axis=1) - tc * turnover.fillna(0.0)

    equity = (1.0 + port_ret).cumprod() * float(capital)
    curve = pd.DataFrame({"ret": port_ret, "equity": equity})

    # Stats
    n = len(port_ret)
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (252.0 / max(1, n)) - 1.0
    sharpe = (
        (port_ret.mean() / port_ret.std(ddof=0) * np.sqrt(252.0))
        if port_ret.std(ddof=0) > 0
        else 0.0
    )

    roll_max = equity.cummax()
    drawdown = equity / roll_max - 1.0
    max_dd = float(drawdown.min())
    calmar = (cagr / abs(max_dd)) if max_dd < 0 else np.nan

    avg_turnover = float(turnover[turnover > 0].mean()) if (turnover > 0).any() else 0.0

    stats = {
        "CAGR": float(cagr),
        "Sharpe": float(sharpe),
        "MaxDD": float(max_dd),
        "Calmar": float(calmar) if not np.isnan(calmar) else 0.0,
        "AvgTurnover": avg_turnover,
    }

    return BacktestResult(curve=curve, weights=weights, stats=stats)

# src/trading_stack_py/portfolio.py
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data_loader import get_prices


@dataclass
class RotationParams:
    lookback: int = 126
    top_n: int = 5
    cost_bps: float = 10.0  # applied on turnover at rebalance


def _month_ends(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    # pick the last available date of each (year, month) in the index
    s = pd.Series(idx, index=idx)
    me = s.groupby([idx.year, idx.month]).max()
    return pd.DatetimeIndex(me.values)


def _align_close_panel(
    tickers: list[str],
    start: str | None,
    end: str | None,
    source: str,
    force_refresh: bool,
) -> pd.DataFrame:
    frames = []
    for t in tickers:
        df = get_prices(
            t, start=start, end=end, source=source, force_refresh=force_refresh
        )
        c = df[["Date", "Close"]].copy()
        c["Date"] = pd.to_datetime(c["Date"])
        c = c.set_index("Date").rename(columns={"Close": t})
        frames.append(c)
    panel = pd.concat(frames, axis=1).sort_index()
    panel = panel.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    return panel


def _momentum(panel_close: pd.DataFrame, lookback: int) -> pd.DataFrame:
    return panel_close / panel_close.shift(lookback) - 1.0


def _equal_weight(names: Iterable[str], chosen: Iterable[str]) -> pd.Series:
    names = list(names)
    chosen = list(chosen)
    if not chosen:
        return pd.Series(0.0, index=names, dtype=float)
    w = 1.0 / float(len(chosen))
    return pd.Series({n: (w if n in chosen else 0.0) for n in names}, dtype=float)


def _build_returns(
    panel_close: pd.DataFrame,
    lookback: int,
    top_n: int,
    cost_bps: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    me = _month_ends(panel_close.index)
    asset_ret = panel_close.pct_change().fillna(0.0)
    mom = _momentum(panel_close, lookback)

    prev_w = pd.Series(0.0, index=panel_close.columns, dtype=float)
    weights = []
    turnover = []
    trades = []
    is_rebal = []

    for dt in panel_close.index:
        if dt in me and mom.index.get_loc(dt) >= lookback:
            scores = mom.loc[dt].dropna()
            chosen = scores.sort_values(ascending=False).head(top_n).index.tolist()
            w = _equal_weight(panel_close.columns, chosen)
            delta = (w - prev_w).abs()
            weights.append(w)
            turnover.append(float(delta.sum()))
            trades.append(int((delta > 1e-12).sum()))
            is_rebal.append(True)
            prev_w = w
        else:
            weights.append(prev_w)
            turnover.append(0.0)
            trades.append(0)
            is_rebal.append(False)

    wdf = pd.DataFrame(weights, index=panel_close.index, columns=panel_close.columns)
    gross = (wdf.shift(1) * asset_ret).sum(axis=1).fillna(0.0)

    cost_rate = cost_bps / 1e4
    cost = pd.Series(turnover, index=panel_close.index, dtype=float) * cost_rate
    net = gross - cost

    perf = pd.DataFrame(
        {
            "Return": net,
            "Rebalance": pd.Series(is_rebal, index=panel_close.index, dtype=bool),
            "Turnover": pd.Series(turnover, index=panel_close.index, dtype=float),
            "Trades": pd.Series(trades, index=panel_close.index, dtype=int),
        }
    )
    return wdf, perf


def backtest_top_n_rotation(
    tickers: list[str],
    start: str | None = "2015-01-01",
    end: str | None = None,
    lookback: int = 126,
    top_n: int = 5,
    cost_bps: float = 10.0,
    source: str = "auto",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Monthly Top-N momentum rotation (equal weight), with costs on turnover at rebalance.
    Returns DataFrame with: Date, Return, Equity, Rebalance, Turnover, Trades, WeightSum
    """
    panel_close = _align_close_panel(tickers, start, end, source, force_refresh)
    _, perf = _build_returns(panel_close, lookback, top_n, cost_bps)

    equity = (1.0 + perf["Return"].fillna(0.0)).cumprod()
    out = perf.copy()
    out.insert(0, "Date", out.index)
    out["Equity"] = equity
    out["WeightSum"] = 1.0  # equal-weighted fully invested after first rebalance
    out = out.reset_index(drop=True)
    return out

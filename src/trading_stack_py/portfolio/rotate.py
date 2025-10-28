# src/trading_stack_py/portfolio/rotate.py
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data_loader import get_prices
from ..metrics.performance import summarize


@dataclass
class RotationConfig:
    top_n: int = 5
    lookback_days: int = 126  # ~6m momentum
    rebal_freq: str = "ME"  # only ME supported for now
    cost_bps: float = 10.0
    source: str = "auto"
    force_refresh: bool = False


def _ensure_date(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"])
        out = out.sort_values("Date").reset_index(drop=True)
    else:
        # if they passed a DatetimeIndex
        out = out.copy()
        out = out.reset_index().rename(columns={"index": "Date"})
        out["Date"] = pd.to_datetime(out["Date"])
        out = out.sort_values("Date").reset_index(drop=True)
    return out


def _close_series(df: pd.DataFrame, ticker: str) -> pd.Series:
    d = _ensure_date(df)
    s = d.set_index("Date")["Close"].astype(float)
    s.name = ticker
    return s


def _build_close_panel(
    tickers: Sequence[str],
    start: str | None,
    end: str | None,
    source: str,
    force_refresh: bool,
) -> pd.DataFrame:
    closes: list[pd.Series] = []
    for t in tickers:
        df = get_prices(
            t, start=start, end=end, source=source, force_refresh=force_refresh
        )
        closes.append(_close_series(df, t))
    # outer join on all dates, then ffill gaps (different holiday calendars)
    panel = pd.concat(closes, axis=1).sort_index()
    panel = panel.ffill()
    return panel


def _month_end_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    ser = pd.Series(index=idx, data=idx)
    me = ser[ser.dt.is_month_end]
    # If the first date is not month end, ensure we start at the first available month-end after start
    return pd.DatetimeIndex(me)


def _turnover(target_w: pd.Series, prev_w: pd.Series) -> float:
    # both aligned over same assets; missing -> 0
    aligned = pd.concat([target_w.fillna(0.0), prev_w.fillna(0.0)], axis=1)
    aligned.columns = ["tgt", "prev"]
    # turnover is L1 distance / 2 if you want buys+sells counted once; we’ll use full L1 as cost applies on both sides.
    return float(np.abs(aligned["tgt"] - aligned["prev"]).sum())


def backtest_top_n_rotation(
    tickers: Sequence[str],
    start: str | None = None,
    end: str | None = None,
    *,
    top_n: int = 5,
    lookback_days: int = 126,
    rebal_freq: str = "ME",
    cost_bps: float = 10.0,
    source: str = "auto",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Equal-weight Top-N (by lookback total return) monthly rotation.
    - Universe: `tickers`
    - Rank at each rebalance on lookback total return
    - Rebalance to equal weights in the winners; apply cost on weight changes
    - Daily portfolio returns between rebalances

    Returns a DataFrame with Date, Equity, Return (+ summary row via metrics.summarize()).
    """
    assert (
        rebal_freq.upper() == "ME"
    ), "Only ME (month-end) rebalance supported currently."

    # 1) Prices panel and daily returns
    prices = _build_close_panel(tickers, start, end, source, force_refresh)
    rets = prices.pct_change().fillna(0.0)
    if len(rets) == 0:
        raise RuntimeError(
            "No return data available for the requested period/universe."
        )

    # 2) Rebalance dates (month-ends where lookback is available)
    me_idx = _month_end_index(prices.index)
    me_idx = me_idx[me_idx >= prices.index.min() + pd.Timedelta(days=lookback_days)]
    if len(me_idx) == 0:
        # Fallback: if tiny span, do a single rebalance at last date
        me_idx = pd.DatetimeIndex([prices.index[-1]])

    # 3) Walk through time, compute weights each rebalance
    port_equity = []
    equity = 1.0
    prev_w = pd.Series(0.0, index=prices.columns)

    # Precompute lookback window prices using shift
    look_close = prices.shift(lookback_days)

    # Rebalance pointer
    next_rebals = set(me_idx)

    # Iterate day-by-day for robust accounting
    for dt in prices.index:
        # Rebalance on month end
        if dt in next_rebals:
            # Momentum = Close / Close[-L] - 1
            mom = (prices.loc[dt] / look_close.loc[dt] - 1.0).replace(
                [np.inf, -np.inf], np.nan
            )
            mom = mom.dropna()
            # pick top N among available
            winners = mom.sort_values(ascending=False).index[
                : max(1, min(top_n, len(mom)))
            ]
            tgt_w = pd.Series(0.0, index=prices.columns)
            if len(winners) > 0:
                tgt_w.loc[winners] = 1.0 / float(len(winners))
            # Turnover / cost
            tv = _turnover(tgt_w, prev_w)  # in weight terms
            cost = tv * (cost_bps / 10000.0)
            prev_w = tgt_w
        else:
            cost = 0.0

        # Daily portfolio return
        day_ret = float((prev_w.fillna(0.0) * rets.loc[dt].fillna(0.0)).sum())
        equity *= 1.0 + day_ret - cost
        port_equity.append((dt, day_ret, cost, equity))

    out = pd.DataFrame(port_equity, columns=["Date", "Return", "Cost", "Equity"])
    out["Date"] = pd.to_datetime(out["Date"])
    out = out.reset_index(drop=True)

    # Summaries in-place (compatible with your summarize())
    out = summarize(out)
    return out

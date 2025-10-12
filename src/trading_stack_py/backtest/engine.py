# src/trading_stack_py/backtest/engine.py
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class _Metrics:
    cagr: float
    sharpe: float
    maxdd: float
    calmar: float


def _compute_metrics(equity: pd.Series, dates: pd.Series) -> _Metrics:
    equity = equity.astype(float)
    rets = equity.pct_change().fillna(0.0)
    n = len(equity)
    if n <= 1:
        return _Metrics(0.0, 0.0, 0.0, 0.0)

    # trading days between first/last date (fallback to 252 if dates missing)
    try:
        years = max(
            (pd.to_datetime(dates.iloc[-1]) - pd.to_datetime(dates.iloc[0])).days / 365.25, 1e-9
        )
        cagr = float((equity.iloc[-1] / max(equity.iloc[0], 1e-12)) ** (1.0 / years) - 1.0)
    except Exception:
        cagr = 0.0

    vol = float(rets.std()) * math.sqrt(252.0)
    sharpe = float((rets.mean() * math.sqrt(252.0)) / vol) if vol > 0 else 0.0

    roll_max = equity.cummax()
    dd = (equity / roll_max) - 1.0
    maxdd = float(dd.min()) if len(dd) else 0.0
    calmar = (cagr / abs(maxdd)) if maxdd < 0 else 0.0

    return _Metrics(cagr, sharpe, maxdd, calmar)


def run_long_only(
    sig: pd.DataFrame,
    entry_col: str = "ENTRY",
    exit_col: str = "EXIT",
    cost_bps: float = 10.0,
) -> pd.DataFrame:
    """
    Executes a simple long-only strategy using boolean ENTRY/EXIT columns.
    Buys 100% on ENTRY when flat; sells 100% on EXIT when long.
    """
    df = sig.copy()
    if "Date" not in df.columns:
        df["Date"] = pd.to_datetime(df.index)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    buy = df[entry_col].astype(bool).fillna(False).to_numpy()
    sell = df[exit_col].astype(bool).fillna(False).to_numpy()
    prices = df["Close"].astype(float).to_numpy()

    cost = float(cost_bps) / 10_000.0
    cash = 1.0
    units = 0.0
    in_pos = False
    trades = 0
    equity_curve = []

    for i in range(len(df)):
        p = prices[i]

        if (not in_pos) and buy[i]:
            # buy all
            units = cash * (1.0 - cost) / p
            cash = 0.0
            in_pos = True
            trades += 1

        elif in_pos and sell[i]:
            # sell all
            cash = units * p * (1.0 - cost)
            units = 0.0
            in_pos = False

        equity_curve.append(cash + units * p)

    out = pd.DataFrame(
        {"Date": df["Date"], "Equity": equity_curve},
        copy=False,
    )

    m = _compute_metrics(out["Equity"], out["Date"])
    out["CAGR"] = np.nan
    out["Sharpe"] = np.nan
    out["MaxDD"] = np.nan
    out["Calmar"] = np.nan
    out["Trades"] = np.nan

    out.iloc[-1, out.columns.get_loc("CAGR")] = m.cagr
    out.iloc[-1, out.columns.get_loc("Sharpe")] = m.sharpe
    out.iloc[-1, out.columns.get_loc("MaxDD")] = m.maxdd
    out.iloc[-1, out.columns.get_loc("Calmar")] = m.calmar
    out.iloc[-1, out.columns.get_loc("Trades")] = trades

    return out

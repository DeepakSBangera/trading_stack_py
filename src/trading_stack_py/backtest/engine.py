from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class _Metrics:
    cagr: float
    sharpe: float
    maxdd: float
    calmar: float
    trades: int


def _annualize_factor(freq: str = "D") -> float:
    # Daily data → ~252 trading days
    return 252.0 if freq.upper().startswith("D") else 252.0


def _safe_series(x: pd.Series) -> pd.Series:
    return x.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)


def _compute_metrics(eq: pd.Series, ret: pd.Series, freq: str = "D") -> _Metrics:
    eq = _safe_series(eq)
    ret = _safe_series(ret)
    ann = _annualize_factor(freq)

    # CAGR
    n = len(eq)
    if n == 0:
        return _Metrics(0.0, 0.0, 0.0, 0.0, 0)
    start, end = float(eq.iloc[0]), float(eq.iloc[-1])
    years = max(1e-9, n / ann)
    cagr = (end / max(1e-9, start)) ** (1.0 / years) - 1.0

    # Sharpe (risk-free ~ 0)
    mu = float(ret.mean()) * ann
    sd = float(ret.std(ddof=0)) * np.sqrt(ann)
    sharpe = mu / sd if sd > 0 else 0.0

    # Max drawdown
    roll_max = eq.cummax()
    dd = eq / roll_max - 1.0
    maxdd = float(dd.min()) if len(dd) else 0.0

    calmar = (cagr / abs(maxdd)) if maxdd < 0 else 0.0
    return _Metrics(cagr, sharpe, maxdd, calmar, 0)


def run_long_only(
    sig_df: pd.DataFrame,
    entry_col: str = "ENTRY",
    exit_col: str = "EXIT",
    cost_bps: float = 10.0,
    price_col: str = "Close",
    date_col: str = "Date",
    freq: str = "D",
) -> pd.DataFrame:
    """
    Long-only, next-bar execution:
      - Signals at bar t are executed at bar t+1 (lagged).
      - Costs applied whenever position changes (both entry & exit).
    Emits columns: Date, Close, Position, Return, Equity, Trades and
    summary metrics (CAGR, Sharpe, MaxDD, Calmar, Trades) on last row only.
    """
    if sig_df is None or len(sig_df) == 0:
        return pd.DataFrame(
            columns=[
                date_col,
                price_col,
                "Position",
                "Return",
                "Equity",
                "Trades",
                "CAGR",
                "Sharpe",
                "MaxDD",
                "Calmar",
            ]
        )

    df = sig_df.copy()

    # Sort & keep only what we need (avoid unused var and be defensive)
    ordered = (date_col, price_col, entry_col, exit_col)
    existing = [c for c in ordered if c in df.columns]
    df = df.loc[:, existing]
    df = df.sort_values(date_col).reset_index(drop=True)
    return df

    # Ensure boolean signals
    entry = df[entry_col].fillna(False).astype(bool)
    exit_ = df[exit_col].fillna(False).astype(bool)

    # Next-bar execution: shift signals by 1 bar
    entry_n = entry.shift(1, fill_value=False)
    exit_n = exit_.shift(1, fill_value=False)

    # Build position path (0 or 1) respecting entries/exits
    pos = np.zeros(len(df), dtype=float)
    in_pos = 0.0
    trades = np.zeros(len(df), dtype=int)
    for i in range(len(df)):
        if exit_n.iloc[i] and in_pos > 0:
            in_pos = 0.0
        elif entry_n.iloc[i] and in_pos == 0:
            in_pos = 1.0

        pos[i] = in_pos
        if i > 0 and pos[i] != pos[i - 1]:
            trades[i] = trades[i - 1] + 1
        elif i > 0:
            trades[i] = trades[i - 1]

    # Price returns
    px = df[price_col].astype(float)
    ret = px.pct_change().fillna(0.0)

    # Strategy return uses previous position (next-bar exec)
    strat_ret_gross = ret * pd.Series(pos, index=df.index).shift(0).fillna(0.0)

    # Costs: whenever position changes today vs yesterday, charge cost
    toggle = np.abs(pd.Series(pos).diff().fillna(0.0))  # 1 on enter, 1 on exit
    cost = (cost_bps / 10_000.0) * toggle
    strat_ret = strat_ret_gross - cost

    equity = (1.0 + strat_ret).cumprod()

    out = pd.DataFrame(
        {
            date_col: df[date_col],
            price_col: px,
            "Position": pos,
            "Return": strat_ret,
            "Equity": equity,
            "Trades": trades,
        }
    )

    # Summary metrics (only last row populated)
    m = _compute_metrics(out["Equity"], out["Return"], freq=freq)
    # overwrite the final row with metrics / trades count
    out.loc[out.index[-1], ["CAGR", "Sharpe", "MaxDD", "Calmar", "Trades"]] = [
        m.cagr,
        m.sharpe,
        m.maxdd,
        m.calmar,
        int(trades[-1]),
    ]

    return out

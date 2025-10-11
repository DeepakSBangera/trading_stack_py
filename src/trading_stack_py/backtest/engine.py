# src/trading_stack_py/backtest/engine.py
from __future__ import annotations

import pandas as pd


def run_long_only(
    df: pd.DataFrame,
    entry_col: str = "ENTRY",
    exit_col: str = "EXIT",
    cost_bps: float = 10.0,  # round-turn estimated cost in basis points
) -> pd.DataFrame:
    """Vectorized long-only backtest using Close-to-Close fills."""
    out = df.copy().sort_values("Date").reset_index(drop=True)

    pos = 0
    cash = 1.0
    units = 0.0
    cost = cost_bps / 10000.0

    curve = []
    for _, row in out.iterrows():
        price = float(row["Close"])
        entry = bool(row.get(entry_col, False))
        exit_ = bool(row.get(exit_col, False))

        if entry and pos == 0:
            # buy all
            units = (cash * (1.0 - cost)) / price
            cash = 0.0
            pos = 1
        elif exit_ and pos == 1:
            # sell all
            cash = units * price * (1.0 - cost)
            units = 0.0
            pos = 0

        equity = cash + units * price
        curve.append(equity)

    out["Equity"] = curve
    out["Return"] = out["Equity"].pct_change().fillna(0.0)
    return out

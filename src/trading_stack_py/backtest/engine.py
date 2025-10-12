from __future__ import annotations

import numpy as np
import pandas as pd


def run_long_only(
    sig: pd.DataFrame,
    entry_col: str = "ENTRY",
    exit_col: str = "EXIT",
    cost_bps: float = 10.0,
) -> pd.DataFrame:
    """
    Very simple long-only backtest:
    - Enter on ENTRY True, exit on EXIT True
    - 100% allocation when in-position, 0% otherwise
    - Apply costs on transitions (entry & exit)
    Returns a frame with Equity and Return columns plus running metrics at the last row.
    """
    df = sig.copy()
    df = df.reset_index(drop=True)

    # Position: 1 in-market, 0 out-of-market
    pos = np.zeros(len(df), dtype=float)
    in_pos = False
    for i in range(len(df)):
        if not in_pos and bool(df.at[i, entry_col]):
            in_pos = True
        elif in_pos and bool(df.at[i, exit_col]):
            in_pos = False
        pos[i] = 1.0 if in_pos else 0.0
    df["Position"] = pos

    # Price-based returns (fallback if 'Close' missing -> zeros)
    if "Close" in df.columns:
        raw_ret = pd.Series(df["Close"]).astype(float).pct_change().fillna(0.0)
    else:
        raw_ret = pd.Series(np.zeros(len(df)))

    # Trading costs: apply when position changes (entry or exit)
    pos_shift = pd.Series(pos).shift(1).fillna(0.0)
    traded = (pd.Series(pos) != pos_shift).astype(float)
    # cost applied on the day of the switch
    cost = traded * (cost_bps / 1e4)

    # Strategy return: position * raw - cost
    strat_ret = pos * raw_ret - cost
    df["Return"] = strat_ret

    # Equity curve (start at 1.0)
    df["Equity"] = (1.0 + df["Return"]).cumprod()

    # Basic running trade count (entries)
    trades = int(pd.Series(df[entry_col]).fillna(False).astype(bool).sum())
    df["Trades"] = trades

    # (Optional) You may compute rolling metrics here and store at last row if you like.
    return df

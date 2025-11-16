# Vectorized long-only backtest with costs, exits, and gates
import numpy as np
import pandas as pd

from .metrics import cagr, calmar, max_drawdown, profit_factor, sharpe, sortino


def _trailing_stop(close: pd.Series, atr: pd.Series, mult: float) -> pd.Series:
    # Simple ATR trailing stop (long): highest(close - mult*atr) since entry
    # Here we approximate with a rolling max of (close - mult*atr)
    base = close - mult * atr
    return base.cummax()


def backtest_symbol(prices: pd.DataFrame, sig: pd.DataFrame, cfg: dict, bt: dict) -> dict:
    """
    prices: [open, high, low, close, volume] indexed by date
    sig:    features + columns [buy, score, sma_f, sma_s, atr, rsi, atr_pct]
    cfg:    signal params (not used here, but passed through)
    bt:     backtest params: tcost_bps, slippage_bps, periods_per_year, use_atr_stop, atr_stop_mult
    """
    if prices.empty or sig.empty:
        return {"daily": pd.DataFrame(), "summary": {"ok": False, "reason": "no_data"}}

    close = prices["close"].astype(float)
    ret = close.pct_change().fillna(0.0)

    # Entry rule: the day's signal 'buy' (1) -> position next day
    buy_flag = sig.get("buy", pd.Series(0, index=prices.index)).fillna(0).astype(int)

    # Exit rule: lose trend (sma_f <= sma_s); optional ATR trailing stop breach
    trend_up = (sig.get("sma_f", close).astype(float) > sig.get("sma_s", close).astype(float)).astype(int)

    # Position logic: once buy fires, hold while trend_up==1; exit when trend_up==0
    held = buy_flag.replace(0, np.nan).ffill().fillna(0).astype(int)
    pos = (held & trend_up).astype(int)

    # Optional ATR stop: if enabled, exit when close < trailing stop
    if bt.get("use_atr_stop", False):
        atr = sig.get("atr", pd.Series(np.nan, index=prices.index)).astype(float)
        trail = _trailing_stop(close, atr, float(bt.get("atr_stop_mult", 3.0)))
        stop_exit = (close < trail).astype(int)  # 1 when below stop
        # clear position when stop hit
        pos = pos.copy()
        pos[stop_exit == 1] = 0
        # ensure no negative values after combining
        pos = (pos > 0).astype(int)

    # Enforce next-day entry/exit effect
    pos_exec = pos.shift(1).fillna(0)

    # Costs: turnover * (tcost + slippage) in bps
    turnover = (pos_exec - pos_exec.shift(1).fillna(0)).abs()
    cost_bps = float(bt.get("tcost_bps", 10)) + float(bt.get("slippage_bps", 5))
    cost = turnover * (cost_bps / 1e4)

    gross = pos_exec * ret
    net = gross - cost

    equity = (1.0 + net).cumprod()
    summ = {}
    ppyr = int(bt.get("periods_per_year", 252))
    summ["n_days"] = int(len(net))
    summ["years"] = float(len(net) / max(ppyr, 1))
    summ["gross_cagr"] = cagr((1 + gross).cumprod(), ppyr)
    summ["net_cagr"] = cagr(equity, ppyr)
    summ["sharpe"] = sharpe(net, ppyr)
    summ["sortino"] = sortino(net, ppyr)
    summ["mdd"] = max_drawdown(equity)
    summ["profit_factor"] = profit_factor(net)
    summ["calmar"] = calmar(summ["net_cagr"], summ["mdd"])
    summ["turnover_year"] = float(turnover.sum() / max(summ["years"], 1e-9))

    daily = pd.DataFrame({"ret": ret, "gross": gross, "net": net, "equity": equity, "pos": pos_exec})
    return {"daily": daily, "summary": summ}


def summarize_universe(results: dict, gates: dict, ppyr: int) -> pd.DataFrame:
    rows = []
    for sym, res in results.items():
        s = res["summary"]
        s["symbol"] = sym
        rows.append(s)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Gate checks
    df["pass_years"] = df["years"] >= float(gates.get("min_years", 1.0))
    df["pass_cagr"] = df["net_cagr"] >= float(gates.get("min_cagr", 0.10))
    df["pass_mdd"] = df["mdd"] >= float(gates.get("max_mdd", -0.25))
    df["pass_sharpe"] = df["sharpe"] >= float(gates.get("min_sharpe", 1.0))
    df["pass_pf"] = df["profit_factor"] >= float(gates.get("min_profit_factor", 1.3))
    df["pass_all"] = df[["pass_years", "pass_cagr", "pass_mdd", "pass_sharpe", "pass_pf"]].all(axis=1)
    return df.sort_values(by=["pass_all", "net_cagr", "sharpe"], ascending=[False, False, False])

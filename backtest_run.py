# Backtest all symbols in watchlist, write summaries and daily equity
import os, yaml
import pandas as pd
from src import data_io
from src.signals import make_signals
from src.backtest import backtest_symbol, summarize_universe

CFG = yaml.safe_load(open("config/config.yaml","r",encoding="utf-8"))
ppyr = int(CFG.get("backtest", {}).get("periods_per_year", 252))
bt = CFG.get("backtest", {})
gates = CFG.get("gate", {})
params = CFG.get("signals", {}).get("params", {})
rule = CFG.get("signals", {}).get("rule", "R1_trend_breakout_obv")

os.makedirs("reports", exist_ok=True)

symbols = data_io.load_watchlist(CFG["data"]["universe_csv"])
source  = CFG["data"].get("source","csv")
start   = CFG["data"].get("start","2018-01-01")
csv_dir = CFG["data"].get("csv_dir","data/csv")

results, daily_eq = {}, []

for sym in symbols:
    # load data
    if source == "csv":
        prices = data_io.fetch_ohlcv_from_csv(sym, csv_dir=csv_dir)
    else:
        prices = data_io.fetch_ohlcv(sym, start=start)
    if prices.empty:
        print(f"[WARN] No data for {sym}")
        results[sym] = {"daily": pd.DataFrame(), "summary": {"ok": False, "reason": "no_data"}}
        continue

    # signals across full history
    sig = make_signals(prices, params, rule)
    res = backtest_symbol(prices, sig, params, bt)
    results[sym] = res

    if not res["daily"].empty:
        d = res["daily"][["net","equity"]].copy()
        d["symbol"] = sym
        d.index.name = "date"
        daily_eq.append(d.reset_index())

# Write per-symbol summary
summ = summarize_universe(results, gates, ppyr)
summ_path = "reports/backtest_summary.csv"
summ.to_csv(summ_path, index=False)
print(f"Wrote {summ_path}")

# Equal-weight portfolio equity (only symbols with data)
if daily_eq:
    df_all = pd.concat(daily_eq, ignore_index=True).set_index("date")
    # pivot to wide returns, average across symbols daily (equal-weight)
    wide = df_all.pivot_table(index=df_all.index, columns="symbol", values="net").sort_index()
    wide = wide.fillna(0.0)
    port_ret = wide.mean(axis=1)
    port_eq = (1 + port_ret).cumprod()
    port = pd.DataFrame({"ret": port_ret, "equity": port_eq})
    port_path = "reports/backtest_daily_equity.csv"
    port.to_csv(port_path)
    print(f"Wrote {port_path}")

# Gate decision (portfolio-level quick check)
if daily_eq:
    from src.metrics import cagr, sharpe, max_drawdown, profit_factor, calmar
    port_cagr = cagr(port_eq, ppyr)
    port_sharpe = sharpe(port_ret, ppyr)
    port_mdd = max_drawdown(port_eq)
    port_pf = profit_factor(port_ret)
    port_calmar = calmar(port_cagr, port_mdd)
    print(f"Portfolio — CAGR: {port_cagr:.2%}, Sharpe: {port_sharpe:.2f}, MDD: {port_mdd:.2%}, PF: {port_pf:.2f}, Calmar: {port_calmar:.2f}")

    ok = True
    if port_cagr < float(gates.get("min_cagr", 0.10)): ok = False
    if port_mdd < float(gates.get("max_mdd", -0.25)): ok = False
    if port_sharpe < float(gates.get("min_sharpe", 1.0)): ok = False
    if port_pf < float(gates.get("min_profit_factor", 1.3)): ok = False
    print("GO" if ok else "NO-GO")

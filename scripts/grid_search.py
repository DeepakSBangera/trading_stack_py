import argparse
import itertools
import os

import pandas as pd
from src.trading_stack_py.data_loader import get_prices

from src.trading_stack_py.backtest.engine import run_long_only
from src.trading_stack_py.metrics.performance import summarize


def param_signal(df, fast: int, slow: int):
    # reuse your signal with overrides if supported; else simple MA cross here
    px = df.copy()
    px["MA_F"] = px["Close"].rolling(fast, min_periods=fast).mean()
    px["MA_S"] = px["Close"].rolling(slow, min_periods=slow).mean()
    px["LONG"] = px["MA_F"] > px["MA_S"]
    px["ENTRY"] = px["LONG"] & ~px["LONG"].shift(1, fill_value=False)
    px["EXIT"] = ~px["LONG"] & px["LONG"].shift(1, fill_value=False)
    return px


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--cost_bps", type=float, default=12.0)
    ap.add_argument("--source", choices=["auto", "local", "yahoo", "synthetic"], default="auto")
    ap.add_argument("--fast", default="5,10,20,30")
    ap.add_argument("--slow", default="50,100,150,200")
    args = ap.parse_args()

    df = get_prices(args.ticker, start=args.start, end=args.end, source=args.source)
    fasts = [int(x) for x in str(args.fast).split(",")]
    slows = [int(x) for x in str(args.slow).split(",")]

    rows = []
    for f, s in itertools.product(fasts, slows):
        if f >= s:
            continue
        sig = param_signal(df, f, s)
        bt = run_long_only(sig, cost_bps=args.cost_bps)
        stats = summarize(bt)
        rows.append({"fast": f, "slow": s, **{k: float(v) for k, v in stats.items()}})

    os.makedirs("reports", exist_ok=True)
    out = pd.DataFrame(rows).sort_values(["Sharpe", "CAGR"], ascending=[False, False])
    out_path = os.path.join("reports", f"grid_{args.ticker.replace('.', '_')}.csv")
    out.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

import argparse
import os

import pandas as pd

from src.trading_stack_py.backtest.engine import run_long_only
from src.trading_stack_py.data_loader import get_prices
from src.trading_stack_py.metrics.performance import probabilistic_sharpe_ratio, summarize


def ma_signal(df, fast: int, slow: int):
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
    ap.add_argument("--train_years", type=int, default=3)
    ap.add_argument("--test_years", type=int, default=1)
    ap.add_argument("--fast_list", default="10,20,30")
    ap.add_argument("--slow_list", default="100,150,200")
    ap.add_argument("--cost_bps", type=float, default=12.0)
    ap.add_argument("--source", choices=["auto", "local", "yahoo", "synthetic"], default="auto")
    args = ap.parse_args()

    df = get_prices(args.ticker, start=args.start, end=args.end, source=args.source)
    df = df.sort_values("Date").reset_index(drop=True)

    fasts = [int(x) for x in args.fast_list.split(",")]
    slows = [int(x) for x in args.slow_list.split(",")]

    # time split indices
    df["Y"] = pd.to_datetime(df["Date"]).dt.year
    years = sorted(df["Y"].unique())

    rows = []
    i = 0
    while True:
        train_years = years[i : i + args.train_years]
        test_years = years[i + args.train_years : i + args.train_years + args.test_years]
        if len(test_years) == 0:
            break

        tr = df[df["Y"].isin(train_years)].copy()
        te = df[df["Y"].isin(test_years)].copy()
        if len(te) < 50 or len(tr) < 50:
            break

        # pick best params on train by Sharpe
        cand = []
        for f in fasts:
            for s in slows:
                if f >= s:
                    continue
                sig_tr = ma_signal(tr, f, s)
                bt_tr = run_long_only(sig_tr, cost_bps=args.cost_bps)
                st_tr = summarize(bt_tr)
                cand.append((f, s, float(st_tr["Sharpe"]), float(st_tr["CAGR"])))
        if not cand:
            break
        f_best, s_best, _, _ = sorted(cand, key=lambda x: (x[2], x[3]), reverse=True)[0]

        # evaluate on test
        sig_te = ma_signal(te, f_best, s_best)
        bt_te = run_long_only(sig_te, cost_bps=args.cost_bps)
        st_te = summarize(bt_te)

        # PSR vs 0 Sharpe benchmark
        # estimate n as number of daily returns in test
        n = max(2, len(bt_te) - 1)
        sr = float(st_te["Sharpe"])
        psr = probabilistic_sharpe_ratio(sr, 0.0, n, skew=0.0, kurt=3.0)

        rows.append(
            {
                "train_years": f"{train_years[0]}-{train_years[-1]}",
                "test_years": f"{test_years[0]}-{test_years[-1]}",
                "fast": f_best,
                "slow": s_best,
                **{k: float(v) for k, v in st_te.items()},
                "PSR": psr,
            }
        )

        i += args.test_years

    res = pd.DataFrame(rows)
    os.makedirs("reports", exist_ok=True)
    outp = os.path.join("reports", f"walkforward_{args.ticker.replace('.','_')}.csv")
    res.to_csv(outp, index=False)
    print(f"Saved: {outp}")
    if not res.empty:
        print(res.to_string(index=False))
    else:
        print("No walk-forward windows produced results. Check date ranges.")


if __name__ == "__main__":
    main()

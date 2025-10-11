# scripts/runner_cli.py
import argparse
import os

from src.trading_stack_py.backtest.engine import run_long_only
from src.trading_stack_py.data_loader import get_prices
from src.trading_stack_py.metrics.performance import summarize
from src.trading_stack_py.signals.core_signals import basic_long_signal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True, help="e.g., RELIANCE.NS or NIFTYBEES.NS")
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--cost_bps", type=float, default=12.0)
    ap.add_argument(
        "--force_refresh", action="store_true", help="ignore cache and refetch (yahoo/local)"
    )
    ap.add_argument(
        "--source",
        choices=["auto", "local", "yahoo", "synthetic"],
        default="auto",
        help="data source preference (default: auto)",
    )
    args = ap.parse_args()

    df = get_prices(
        args.ticker,
        start=args.start,
        end=args.end,
        force_refresh=args.force_refresh,
        source=args.source,
    )
    sig = basic_long_signal(df)
    bt = run_long_only(sig, cost_bps=args.cost_bps)
    stats = summarize(bt)

    os.makedirs("reports", exist_ok=True)
    out_csv = os.path.join("reports", f"run_{args.ticker.replace('.','_')}.csv")
    bt.to_csv(out_csv, index=False)

    print({"ticker": args.ticker, **{k: round(v, 4) for k, v in stats.items()}})
    print(f"CSV: {out_csv}")


if __name__ == "__main__":
    main()

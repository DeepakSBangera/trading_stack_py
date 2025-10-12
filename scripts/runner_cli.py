# scripts/runner_cli.py
from __future__ import annotations

import argparse
from pathlib import Path

from src.trading_stack_py.backtest.engine import run_long_only
from src.trading_stack_py.config import get_portfolio_params
from src.trading_stack_py.data_loader import get_prices
from src.trading_stack_py.signals.core_signals import basic_long_signal


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--start", required=False, default="2015-01-01")
    ap.add_argument("--end", required=False, default=None)
    ap.add_argument(
        "--cost_bps", type=float, required=False, default=get_portfolio_params().get("cost_bps", 10)
    )
    ap.add_argument(
        "--source",
        choices=["auto", "local", "yahoo", "synthetic"],
        default="auto",
        help="Price source",
    )
    ap.add_argument(
        "--use_crossover",
        action="store_true",
        help="If set: use fast>slow SMA crossover instead of Close>fast SMA",
    )
    ap.add_argument("--force_refresh", action="store_true")
    args = ap.parse_args()

    df = get_prices(
        args.ticker,
        start=args.start,
        end=args.end,
        source=args.source,
        force_refresh=args.force_refresh,
    )

    sig = basic_long_signal(df, use_crossover=args.use_crossover)

    bt = run_long_only(
        sig,
        entry_col="ENTRY",
        exit_col="EXIT",
        cost_bps=args.cost_bps,
    )

    out = Path("reports") / f"run_{args.ticker.replace('.','_')}.csv"
    Path("reports").mkdir(exist_ok=True)
    bt.to_csv(out, index=False)

    # console line
    last = bt.iloc[-1]
    stats = {
        "ticker": args.ticker,
        "CAGR": round(float(last.get("CAGR", 0.0)), 4),
        "Sharpe": round(float(last.get("Sharpe", 0.0)), 4),
        "MaxDD": last.get("MaxDD", 0.0),
        "Calmar": last.get("Calmar", 0.0),
        "Trades": int(last.get("Trades", 0)),
    }
    print(stats)
    print(f"CSV: {out}")


if __name__ == "__main__":
    main()

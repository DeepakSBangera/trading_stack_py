# scripts/runner_cli.py
from __future__ import annotations

import argparse
from pathlib import Path

from src.trading_stack_py.backtest.engine import run_long_only
from src.trading_stack_py.config import load_strategy_config
from src.trading_stack_py.data_loader import get_prices
from src.trading_stack_py.signals.core_signals import basic_long_signal


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run a quick long-only backtest for a single ticker.")
    ap.add_argument("--ticker", required=True, help="Symbol (e.g., RELIANCE.NS)")
    ap.add_argument("--start", default="2015-01-01", help="Start date (YYYY-MM-DD)")
    ap.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")

    # Don’t read config here; allow override via CLI. If None, we’ll pull from config after parse.
    ap.add_argument(
        "--cost_bps",
        type=float,
        default=None,
        help="Round-trip cost in basis points (default from config if omitted).",
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
        help="Force SMA(fast) > SMA(slow) at runtime (overrides config).",
    )
    ap.add_argument(
        "--force_refresh",
        action="store_true",
        help="Ignore cached/local data when using remote sources.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    # Load config now (provides MA windows, default costs, etc.)
    cfg = load_strategy_config()
    # pick cost_bps: CLI override > config > fallback 10
    cfg_cost = float(cfg.portfolio.get("cost_bps", 10))
    cost_bps = float(args.cost_bps) if args.cost_bps is not None else cfg_cost

    # Pull prices
    df = get_prices(
        args.ticker,
        start=args.start,
        end=args.end,
        source=args.source,
        force_refresh=args.force_refresh,
    )

    # If --use_crossover set, override; else None = use config default
    use_crossover_effective = True if args.use_crossover else None

    # Build signal dataframe
    sig = basic_long_signal(df, use_crossover=use_crossover_effective, cfg=cfg)

    # Run backtest
    bt = run_long_only(
        sig,
        entry_col="ENTRY",
        exit_col="EXIT",
        cost_bps=cost_bps,
    )

    # Write results
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    out = reports_dir / f"run_{args.ticker.replace('.', '_')}.csv"
    bt.to_csv(out, index=False)

    # Console summary
    last = bt.iloc[-1] if len(bt) else {}
    stats = {
        "ticker": args.ticker,
        "CAGR": round(float(last.get("CAGR", 0.0)), 4) if last else 0.0,
        "Sharpe": round(float(last.get("Sharpe", 0.0)), 4) if last else 0.0,
        "MaxDD": float(last.get("MaxDD", 0.0)) if last else 0.0,
        "Calmar": float(last.get("Calmar", 0.0)) if last else 0.0,
        "Trades": int(last.get("Trades", 0)) if last else 0,
    }
    print(stats)
    print(f"CSV: {out}")


if __name__ == "__main__":
    main()

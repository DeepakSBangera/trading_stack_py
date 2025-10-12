# scripts/runner_cli.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from src.trading_stack_py.backtest.engine import run_long_only
from src.trading_stack_py.config import load_strategy_config
from src.trading_stack_py.data_loader import get_prices
from src.trading_stack_py.signals.core_signals import basic_long_signal


def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        # Handle numpy types, pandas scalars, etc.
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument(
        "--cost_bps",
        type=float,
        default=None,  # pick from config if available
        help="Round-trip cost in basis points for each trade",
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
        help="If set: use SMA(fast) > SMA(slow). If not set, default is Close > SMA(fast), unless config says otherwise.",
    )
    ap.add_argument("--force_refresh", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    # Load YAML strategy config (safe defaults if file/fields missing)
    cfg = load_strategy_config()

    # Effective cost_bps: CLI overrides config; fallback to 10
    cfg_cost = getattr(getattr(cfg, "portfolio", None), "cost_bps", None)
    cost_bps = (
        args.cost_bps if args.cost_bps is not None else (cfg_cost if cfg_cost is not None else 10.0)
    )

    # Effective crossover flag: CLI True forces crossover; otherwise use config flag (default False)
    cfg_use_x: bool | None = getattr(getattr(cfg, "signals", None), "use_crossover", None)
    use_crossover_effective = True if args.use_crossover else bool(cfg_use_x or False)

    # Pull prices
    df = get_prices(
        args.ticker,
        start=args.start,
        end=args.end,
        source=args.source,
        force_refresh=args.force_refresh,
    )

    # Build signal using config (windows) + runtime crossover override
    sig = basic_long_signal(df, use_crossover=use_crossover_effective, cfg=cfg)

    # Backtest
    bt = run_long_only(sig, entry_col="ENTRY", exit_col="EXIT", cost_bps=cost_bps)

    # Save CSV
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    out_path = reports_dir / f"run_{args.ticker.replace('.', '_')}.csv"
    bt.to_csv(out_path, index=False)

    # Print compact stats (safe even if DataFrame empty or columns missing)
    if bt.empty:
        stats = {
            "ticker": args.ticker,
            "CAGR": 0.0,
            "Sharpe": 0.0,
            "MaxDD": 0.0,
            "Calmar": 0.0,
            "Trades": 0,
        }
    else:
        last = bt.iloc[-1]
        stats = {
            "ticker": args.ticker,
            "CAGR": round(_to_float(last.get("CAGR", 0.0)), 4),
            "Sharpe": round(_to_float(last.get("Sharpe", 0.0)), 4),
            "MaxDD": _to_float(last.get("MaxDD", 0.0)),
            "Calmar": _to_float(last.get("Calmar", 0.0)),
            "Trades": int(_to_float(last.get("Trades", 0))),
        }

    print(stats)
    print(f"CSV: {out_path}")


if __name__ == "__main__":
    main()

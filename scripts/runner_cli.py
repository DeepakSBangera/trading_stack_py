# scripts/runner_cli.py
from __future__ import annotations

import argparse
from pathlib import Path

from src.trading_stack_py.data_loader import get_prices
from src.trading_stack_py.signals.core_signals import basic_long_signal

from src.trading_stack_py.backtest.engine import run_long_only
from src.trading_stack_py.config import load_strategy_config


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--cost_bps", type=float, default=10.0)
    ap.add_argument(
        "--source",
        choices=["auto", "local", "yahoo", "synthetic"],
        default="auto",
    )
    ap.add_argument(
        "--use_crossover",
        action="store_true",
        help="Force SMA(fast)>SMA(slow) instead of Close>SMA(fast)",
    )
    ap.add_argument("--force_refresh", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_strategy_config()  # reads config/strategy.yml|yaml

    # read MA & signal defaults from YAML
    ma_cfg = getattr(cfg, "ma", {}) or {}
    fast = int(ma_cfg.get("fast", 20))
    slow = int(ma_cfg.get("slow", 100))

    sig_cfg = getattr(cfg, "signals", {}) or {}
    use_crossover_effective = bool(sig_cfg.get("use_crossover", False)) or args.use_crossover

    df = get_prices(
        args.ticker,
        start=args.start,
        end=args.end,
        source=args.source,
        force_refresh=args.force_refresh,
    )

    sig = basic_long_signal(df, use_crossover=use_crossover_effective, fast=fast, slow=slow)

    bt = run_long_only(sig, entry_col="ENTRY", exit_col="EXIT", cost_bps=args.cost_bps)

    reports = Path("reports")
    reports.mkdir(exist_ok=True)
    out = reports / f"run_{args.ticker.replace('.', '_')}.csv"
    bt.to_csv(out, index=False)

    sig_out = reports / f"signal_{args.ticker.replace('.', '_')}.csv"
    sig.to_csv(sig_out, index=False)

    last = bt.iloc[-1] if not bt.empty else None
    stats = {
        "ticker": args.ticker,
        "CAGR": round(float(last["CAGR"]), 4) if last is not None else 0.0,
        "Sharpe": round(float(last["Sharpe"]), 4) if last is not None else 0.0,
        "MaxDD": float(last["MaxDD"]) if last is not None else 0.0,
        "Calmar": float(last["Calmar"]) if last is not None else 0.0,
        "Trades": int(last["Trades"]) if last is not None else 0,
        "entries": int(sig["ENTRY"].sum()),
        "exits": int(sig["EXIT"].sum()),
        "first_date": str(sig["Date"].iloc[0]),
        "last_date": str(sig["Date"].iloc[-1]),
        "signal_csv": str(sig_out),
    }
    print(stats)
    print(f"CSV: {out}")


if __name__ == "__main__":
    main()

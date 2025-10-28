from __future__ import annotations

import argparse
from pathlib import Path

from .backtest.engine import run_long_only
from .config import load_strategy_config
from .data_loader import get_prices
from .metrics.performance import summarize
from .signals.core_signals import basic_long_signal


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="trading-stack",
        description="Run a single-symbol backtest.",
    )
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--cost_bps", type=float, default=10.0)
    ap.add_argument(
        "--source",
        choices=["auto", "local", "yahoo", "synthetic"],
        default="auto",
        help="Price source",
    )
    ap.add_argument(
        "--use_crossover",
        action="store_true",
        help="Use fast>slow SMA crossover (overrides YAML signals.use_crossover / use_crossover).",
    )
    ap.add_argument("--force_refresh", action="store_true")
    return ap.parse_args()


def _cfg_use_crossover(cfg) -> bool:
    """
    Be tolerant to config shapes:

    1) Nested:
       signals:
         use_crossover: true

    2) Flat:
       use_crossover: true
    """
    # nested structure
    if hasattr(cfg, "signals") and hasattr(cfg.signals, "use_crossover"):
        try:
            return bool(cfg.signals.use_crossover)
        except Exception:
            pass
    # flat structure
    if hasattr(cfg, "use_crossover"):
        try:
            return bool(cfg.use_crossover)
        except Exception:
            pass
    # safest default
    return False


def main() -> None:
    args = _parse_args()
    cfg = load_strategy_config()

    # runtime flag overrides YAML; otherwise read (nested or flat) from cfg
    use_crossover_effective = True if args.use_crossover else _cfg_use_crossover(cfg)

    df = get_prices(
        args.ticker,
        start=args.start,
        end=args.end,
        source=args.source,
        force_refresh=args.force_refresh,
    )

    sig = basic_long_signal(df, use_crossover=use_crossover_effective)

    bt = run_long_only(
        sig,
        entry_col="ENTRY",
        exit_col="EXIT",
        cost_bps=args.cost_bps,
    )

    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    out = reports_dir / f"run_{args.ticker.replace('.','_')}.csv"
    bt.to_csv(out, index=False)

    # --- Robust stats: prefer last row if present; else compute via summarize(bt)
    want = {"CAGR", "Sharpe", "MaxDD", "Calmar", "Trades"}
    have = set(bt.columns)
    if len(bt) and want.issubset(have):
        last = bt.iloc[-1]
        cagr = float(last.get("CAGR", 0.0))
        sharpe = float(last.get("Sharpe", 0.0))
        maxdd = float(last.get("MaxDD", 0.0))
        calmar = float(last.get("Calmar", 0.0))
        trades = int(last.get("Trades", 0))
    else:
        st = summarize(bt) if len(bt) else {}
        cagr = float(st.get("CAGR", 0.0))
        sharpe = float(st.get("Sharpe", 0.0))
        maxdd = float(st.get("MaxDD", 0.0))
        calmar = float(st.get("Calmar", 0.0))
        trades = int(st.get("Trades", 0))

    # helpful signal breadcrumbs
    sig_out = reports_dir / f"signal_{args.ticker.replace('.','_')}.csv"
    try:
        sig.to_csv(sig_out, index=False)
    except Exception:
        pass

    stats = {
        "ticker": args.ticker,
        "CAGR": round(cagr, 4),
        "Sharpe": round(sharpe, 4),
        "MaxDD": maxdd,
        "Calmar": calmar,
        "Trades": trades,
        "entries": int(sig["ENTRY"].sum()) if "ENTRY" in sig else 0,
        "exits": int(sig["EXIT"].sum()) if "EXIT" in sig else 0,
        "first_date": str(sig["Date"].iloc[0]) if len(sig) and "Date" in sig else "",
        "last_date": str(sig["Date"].iloc[-1]) if len(sig) and "Date" in sig else "",
        "signal_csv": str(sig_out),
    }

    print(stats)
    print(f"CSV: {out}")


if __name__ == "__main__":
    main()

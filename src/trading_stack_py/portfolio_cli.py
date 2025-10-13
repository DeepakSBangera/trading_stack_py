# src/trading_stack_py/portfolio_cli.py
from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from .portfolio import backtest_top_n_rotation


# ----------------------------- Arg parsing ----------------------------- #
def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="trading-stack-portfolio",
        description="Top-N monthly rotation backtest across a ticker universe.",
    )
    ap.add_argument(
        "--tickers",
        required=False,
        help="Comma-separated tickers. If omitted, tries YAML watchlist/universe.",
    )
    ap.add_argument("--start", required=False, default=None)
    ap.add_argument("--end", required=False, default=None)
    ap.add_argument(
        "--lookback", type=int, default=126, help="Momentum lookback days (default: 126)."
    )
    ap.add_argument("--top_n", type=int, default=None, help="Override YAML top_n.")
    ap.add_argument(
        "--source",
        choices=["auto", "local", "yahoo", "synthetic"],
        default="auto",
        help="Price source",
    )
    ap.add_argument("--cost_bps", type=float, default=None, help="Override YAML cost_bps.")
    ap.add_argument("--force_refresh", action="store_true")
    return ap.parse_args()


# ----------------------------- Coercion helpers ----------------------------- #
def _try_df(obj: Any, dates: pd.Series | None = None) -> pd.DataFrame | None:
    """Try to coerce various shapes to a two-column DataFrame (Date, Equity)."""
    # Already a DataFrame?
    if isinstance(obj, pd.DataFrame):
        df = obj.copy()
        if "Date" not in df.columns and len(df.columns) >= 1:
            df.rename(columns={df.columns[0]: "Date"}, inplace=True)
        if "Equity" not in df.columns:
            non_date_cols = [c for c in df.columns if c != "Date"]
            if len(non_date_cols) == 1:
                df.rename(columns={non_date_cols[0]: "Equity"}, inplace=True)
        return df

    # Series → DataFrame
    if isinstance(obj, pd.Series):
        s = obj.copy()
        s.name = s.name or "Equity"
        if dates is not None and len(dates) == len(s):
            return pd.DataFrame({"Date": dates, "Equity": s.values})
        if isinstance(s.index, pd.DatetimeIndex | pd.Index) and len(s.index) == len(s):
            return pd.DataFrame({"Date": s.index, "Equity": s.values})
        return pd.DataFrame({"Date": range(len(s)), "Equity": s.values})

    # list | tuple → DataFrame
    if isinstance(obj, list | tuple) and len(obj) > 0:
        if dates is not None and len(dates) == len(obj):
            return pd.DataFrame({"Date": dates, "Equity": list(obj)})
        return pd.DataFrame({"Date": range(len(obj)), "Equity": list(obj)})

    # dict → attempt common keys / shapes
    if isinstance(obj, dict):
        # Common container keys
        for key in ("equity", "equity_curve", "df", "result"):
            if key in obj:
                return _try_df(obj[key], dates=dates)

        # Direct vectors
        if "Equity" in obj and "Date" in obj:
            return pd.DataFrame({"Date": obj["Date"], "Equity": obj["Equity"]})
        if "equity" in obj and "dates" in obj:
            return pd.DataFrame({"Date": obj["dates"], "Equity": obj["equity"]})

        return None

    return None


def _extract_equity_df(res: Any) -> pd.DataFrame:
    """Extract a DataFrame (Date, Equity) from a variety of result shapes."""
    # Direct DF
    df = _try_df(res)
    if df is not None:
        return df

    # Tuple / List with a DataFrame somewhere
    if isinstance(res, list | tuple):
        for item in res:
            df = _try_df(item)
            if df is not None:
                return df

    # Dict with nested objects
    if isinstance(res, dict):
        for v in res.values():
            df = _try_df(v)
            if df is not None:
                return df

    raise TypeError(
        "Unexpected return type from backtest_top_n_rotation. "
        "Expected DataFrame or a dict/list/tuple containing a DataFrame or (dates, equity) vectors."
    )


def _summarize_result(res: Any, df_fallback: pd.DataFrame | None) -> dict[str, Any]:
    """Return a small dict of printable stats, even if we only got a metrics dict."""
    if isinstance(res, dict):
        keys = ["CAGR", "Sharpe", "MaxDD", "Calmar", "top_n", "lookback_days", "rebal_freq"]
        return {k: res[k] for k in keys if k in res}

    if isinstance(df_fallback, pd.DataFrame) and len(df_fallback) > 0:
        last = df_fallback.iloc[-1]
        out: dict[str, Any] = {}
        for k in ("CAGR", "Sharpe", "MaxDD", "Calmar"):
            if k in df_fallback.columns:
                try:
                    out[k] = float(last.get(k, 0.0))
                except Exception:
                    pass
        return out

    return {}


def _jsonable(o: Any) -> Any:
    """Safe JSON dump for debug writes."""
    try:
        json.dumps(o)
        return o
    except Exception:
        if isinstance(o, pd.DataFrame):
            return {"_type": "DataFrame", "shape": list(o.shape), "cols": list(o.columns)}
        if isinstance(o, pd.Series):
            return {"_type": "Series", "name": o.name, "rows": int(len(o))}
        if isinstance(o, list | tuple):
            return [_jsonable(x) for x in o]
        if isinstance(o, dict):
            return {k: _jsonable(v) for k, v in o.items()}
        return str(o)


def _dump_debug_payload(res: Any, reports: Path) -> Path:
    reports.mkdir(exist_ok=True)
    raw_path = reports / "portfolio_raw.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(_jsonable(res), f, indent=2)
    return raw_path


# ----------------------------- Main ----------------------------- #
def main() -> None:
    args = _parse_args()

    # Tickers
    if args.tickers:
        tickers: Sequence[str] = [t.strip() for t in args.tickers.split(",") if t.strip()]
    else:
        tickers = []

    # Call engine (matches rotate.backtest_top_n_rotation signature)
    res = backtest_top_n_rotation(
        tickers=tickers,
        start=args.start,
        end=args.end,
        top_n=args.top_n if args.top_n is not None else 5,
        lookback_days=int(args.lookback),
        rebal_freq="ME",
        cost_bps=args.cost_bps if args.cost_bps is not None else 10.0,
        source=args.source,
        force_refresh=bool(args.force_refresh),
    )

    # Try to extract equity curve
    reports = Path("reports")
    try:
        df = _extract_equity_df(res)
    except Exception:
        raw_path = _dump_debug_payload(res, reports)
        summary = {
            "type": type(res).__name__,
            "has_keys": list(res.keys()) if isinstance(res, dict) else None,
            "len": len(res) if hasattr(res, "__len__") else None,
            "raw_dump": str(raw_path),
        }
        print("Could not coerce portfolio result into a DataFrame.")
        print(f"Result summary: {summary}")
        print(
            "Wrote raw result for inspection. Please open the file above and share its shape; the CLI will be updated to accept that structure."
        )
        return

    # Write CSV of the equity curve (or metrics DF)
    reports.mkdir(exist_ok=True)
    tag = "-".join([t.replace(".", "_") for t in (tickers if tickers else ["UNIVERSE"])])
    out = reports / f"portfolio_{tag}.csv"
    df.to_csv(out, index=False)

    # Prepare a concise console summary
    metrics = _summarize_result(res, df)
    payload = {
        "tickers": list(tickers),
        "top_n": int(args.top_n) if args.top_n is not None else 5,
        "lookback_days": int(args.lookback),
        "rebal_freq": "ME",
        **metrics,
    }

    print(payload)
    print(f"CSV: {out}")


if __name__ == "__main__":
    main()

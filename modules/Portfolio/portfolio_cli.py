# src/trading_stack_py/portfolio_cli.py
from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from .portfolio import backtest_top_n_rotation


# ---------------------------
# CLI parsing
# ---------------------------
def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="trading-stack-portfolio",
        description="Top-N monthly rotation backtest across a ticker universe.",
    )
    ap.add_argument(
        "--tickers",
        help="Comma-separated tickers. If omitted, tries YAML watchlist/universe.",
    )
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument(
        "--lookback",
        type=int,
        default=126,
        help="Momentum lookback days (default: 126).",
    )
    ap.add_argument("--top_n", type=int, default=None, help="Override YAML top_n.")
    ap.add_argument(
        "--source",
        choices=["auto", "local", "yahoo", "synthetic"],
        default="auto",
        help="Price source",
    )
    ap.add_argument(
        "--cost_bps",
        type=float,
        default=None,
        help="Override YAML cost_bps.",
    )
    ap.add_argument("--force_refresh", action="store_true")
    return ap.parse_args()


# ---------------------------
# Result coercion helpers
# ---------------------------
def _coerce_df_from_like(obj: Any, dates: Iterable[Any] | None = None) -> pd.DataFrame | None:
    """
    Try to coerce various 'equity-like' objects into a DataFrame with Date/Equity columns.
    Returns None if not possible.
    """
    # Already a DataFrame
    if isinstance(obj, pd.DataFrame):
        df = obj.copy()
        if "Date" not in df.columns:
            # If there's an index that looks like dates, move it out
            if isinstance(df.index, pd.DatetimeIndex | pd.Index) and len(df.index) == len(df):
                df = df.reset_index().rename(columns={"index": "Date"})
            else:
                # best effort date
                if "Date" not in df.columns:
                    if isinstance(df.index, pd.DatetimeIndex | pd.Index) and len(df.index) == len(df):
                        df = df.reset_index().rename(columns={"index": "Date"})
                # If first column isn't Date, but looks like dates, rename it
                if "Date" not in df.columns and df.shape[1] >= 2:
                    if isinstance(df.iloc[:, 0], pd.Series) and isinstance(df.index, pd.DatetimeIndex | pd.Index):
                        df.rename(columns={df.columns[0]: "Date"}, inplace=True)
        return df

    # Series -> DataFrame(Date, Equity)
    if isinstance(obj, pd.Series):
        s = obj
        if dates is not None and len(s) == len(list(dates)):
            return pd.DataFrame({"Date": list(dates), "Equity": s.values})
        if isinstance(s.index, pd.DatetimeIndex | pd.Index) and len(s.index) == len(s):
            return pd.DataFrame({"Date": s.index, "Equity": s.values})
        return pd.DataFrame({"Date": range(len(s)), "Equity": s.values})

    # List/Tuple of numbers, with optional parallel dates
    if isinstance(obj, list | tuple) and len(obj) > 0:
        if dates is not None and len(list(dates)) == len(obj):
            return pd.DataFrame({"Date": list(dates), "Equity": list(obj)})
        return pd.DataFrame({"Date": range(len(obj)), "Equity": list(obj)})

    return None


def _extract_equity_df(res: Any) -> pd.DataFrame:
    """
    Accepts several shapes and extracts an equity DataFrame.
    Raises TypeError if no DataFrame-like object is found.
    """
    # If the result is already a DataFrame
    if isinstance(res, pd.DataFrame):
        return _coerce_df_from_like(res)  # type: ignore[return-value]

    # Dict-like containers that may hold the curve
    if isinstance(res, dict):
        for key in ("equity", "equity_curve", "df", "result"):
            if key in res:
                df = _coerce_df_from_like(res[key], res.get("dates"))
                if isinstance(df, pd.DataFrame):
                    if "Date" not in df and "dates" in res:
                        df = df.copy()
                        df.insert(0, "Date", list(res["dates"]))  # type: ignore[index]
                    return df
        # Some engines return metrics only
        raise TypeError("metrics-only")

    # Tuple / List with a DataFrame somewhere
    if isinstance(res, list | tuple):
        for item in res:
            try:
                df = _coerce_df_from_like(item)
                if isinstance(df, pd.DataFrame):
                    return df
            except Exception:
                pass

    # A plain Series / array / list also acceptable
    df2 = _coerce_df_from_like(res)
    if isinstance(df2, pd.DataFrame):
        return df2

    raise TypeError(
        "Unexpected return type from backtest_top_n_rotation. "
        "Expected DataFrame or dict containing a DataFrame under keys: "
        "equity, equity_curve, df, or result."
    )


def _jsonable(obj: Any) -> Any:
    def _json_fallback(o: Any) -> Any:
        if isinstance(o, pd.DataFrame):
            return {
                "_type": "DataFrame",
                "shape": list(o.shape),
                "cols": list(o.columns),
            }
        if isinstance(o, pd.Series):
            return {"_type": "Series", "name": o.name, "rows": int(len(o))}
        if isinstance(o, list | tuple):
            return [_jsonable(x) for x in o]
        if isinstance(o, dict):
            return {str(k): _jsonable(v) for k, v in o.items()}
        return str(o)

    try:
        json.dumps(obj)  # type: ignore[arg-type]
        return obj
    except Exception:
        return _json_fallback(obj)


def _dump_debug_payload(res: Any, reports: Path) -> str:
    reports.mkdir(exist_ok=True)
    path = reports / "portfolio_raw.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_jsonable(res), f, indent=2, default=str)
    return str(path)


# ---------------------------
# Main
# ---------------------------
def main() -> None:
    args = _parse_args()

    tickers: Sequence[str] = []
    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]

    # Map CLI → engine signature
    top_n = args.top_n if args.top_n is not None else 5
    cost_bps = float(args.cost_bps) if args.cost_bps is not None else 10.0
    lookback_days = int(args.lookback)

    # Run engine
    res = backtest_top_n_rotation(
        tickers=tickers,
        start=args.start,
        end=args.end,
        top_n=top_n,
        lookback_days=lookback_days,
        rebal_freq="ME",
        cost_bps=cost_bps,
        source=args.source,
        force_refresh=args.force_refresh,
    )

    reports = Path("reports")
    reports.mkdir(exist_ok=True)
    tag = "-".join([t.replace(".", "_") for t in tickers[:5]])

    # Try to coerce to an equity curve first
    try:
        df = _extract_equity_df(res)
        out = reports / f"portfolio_{tag}.csv"
        df.to_csv(out, index=False)
        # Print a minimal stats summary if present
        last = df.iloc[-1] if len(df) else None
        stats = {
            "tickers": tickers,
            "top_n": int(top_n),
            "lookback_days": int(lookback_days),
            "rebal_freq": "ME",
        }
        # If the engine appended metrics into df (not required), add safely
        if last is not None:
            for k in ("CAGR", "Sharpe", "MaxDD", "Calmar"):
                if k in df.columns:
                    try:
                        stats[k] = round(float(last.get(k, 0.0)), 4)  # type: ignore[attr-defined]
                    except Exception:
                        pass
        print(stats)
        print(f"CSV: {out}")
        return
    except TypeError as e:
        # metrics-only or truly unknown
        if "metrics-only" not in str(e):
            raw_path = _dump_debug_payload(res, reports)
            print(
                "Could not coerce portfolio result into a DataFrame.\n"
                f"Result summary: {{'type': '{type(res).__name__}', "
                f"'has_keys': {list(res.keys()) if isinstance(res, dict) else 'n/a'}, "
                f"'len': {len(res) if hasattr(res, '__len__') else 'n/a'}, "
                f"'raw_dump': '{raw_path}'}}"
            )
            print(
                "Wrote raw result for inspection. Please open the file above and share its shape; the CLI will be updated to accept that structure."
            )
            return

    # Metrics-only path (dict with CAGR/Sharpe/MaxDD/Calmar)
    if isinstance(res, dict):
        metrics_keys = ("CAGR", "Sharpe", "MaxDD", "Calmar")
        has_all = all(k in res for k in metrics_keys)
        if has_all:
            out = reports / f"portfolio_{tag}_metrics.csv"
            pd.DataFrame(
                [
                    {
                        "tickers": ",".join(tickers),
                        "top_n": int(top_n),
                        "lookback_days": int(lookback_days),
                        "rebal_freq": "ME",
                        "CAGR": float(res["CAGR"]),
                        "Sharpe": float(res["Sharpe"]),
                        "MaxDD": float(res["MaxDD"]),
                        "Calmar": float(res["Calmar"]),
                    }
                ]
            ).to_csv(out, index=False)

            print(
                {
                    "tickers": tickers,
                    "top_n": int(top_n),
                    "lookback_days": int(lookback_days),
                    "rebal_freq": "ME",
                    "CAGR": round(float(res["CAGR"]), 4),
                    "Sharpe": round(float(res["Sharpe"]), 4),
                    "MaxDD": float(res["MaxDD"]),
                    "Calmar": float(res["Calmar"]),
                }
            )
            print(f"CSV: {out}")
            return

    # Fallback: dump raw for inspection
    raw_path = _dump_debug_payload(res, reports)
    print(
        "Could not coerce portfolio result into a DataFrame.\n"
        f"Result summary: {{'type': '{type(res).__name__}', "
        f"'has_keys': {list(res.keys()) if isinstance(res, dict) else 'n/a'}, "
        f"'len': {len(res) if hasattr(res, '__len__') else 'n/a'}, "
        f"'raw_dump': '{raw_path}'}}"
    )
    print(
        "Wrote raw result for inspection. Please open the file above and share its shape; the CLI will be updated to accept that structure."
    )


if __name__ == "__main__":
    main()

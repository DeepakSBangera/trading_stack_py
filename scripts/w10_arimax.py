# scripts/w10_arimax.py
"""
W10 — ARIMAX/SARIMAX forecast evaluator.

Reads price CSVs (via glob), fits a baseline SARIMAX(p,d,q) per symbol,
evaluates on a holdout, and writes metrics to reports/wk10_forecast_eval.csv.

Expected CSV columns (flexible):
- Price: one of ["Adj Close", "adj_close", "Close", "close", "price"]
- Date  : optional, one of ["Date", "date", "Timestamp", "timestamp"]

Usage (PowerShell):
  python scripts/w10_arimax.py `
    --data-glob "data/csv/*.csv" `
    --out "reports/wk10_forecast_eval.csv" `
    --order "1,1,1"
"""

from __future__ import annotations

import argparse
import sys
import warnings
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
from statsmodels.tsa.statespace.sarimax import SARIMAX

PRICE_CANDIDATES: tuple[str, ...] = (
    "adj close",
    "adj_close",
    "close",
    "price",
)
DATE_CANDIDATES: tuple[str, ...] = (
    "date",
    "timestamp",
)


def _first_match(candidates: Iterable[str], haystack: Iterable[str]) -> str | None:
    lower = {str(c).lower(): c for c in haystack}
    for want in candidates:
        if want in lower:
            return lower[want]
    return None


def load_price_series(path: Path) -> pd.Series:
    """Load a price series from CSV, return float series indexed by date (if present)."""
    df = pd.read_csv(path)

    # Pick price column (case-insensitive)
    price_col = _first_match(PRICE_CANDIDATES, df.columns)
    if not price_col:
        raise ValueError(f"No price-like column found in {path}")

    # Optional: pick date column and set index
    date_col = _first_match(DATE_CANDIDATES, df.columns)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)

    s = pd.to_numeric(df[price_col], errors="coerce").dropna()

    # Ensure monotonic index if datetime indexed
    if isinstance(s.index, pd.DatetimeIndex) and not s.index.is_monotonic_increasing:
        s = s.sort_index()

    # Give the series a stable frequency if possible (reduces SARIMAX warnings)
    if isinstance(s.index, pd.DatetimeIndex):
        freq = pd.infer_freq(s.index)
        if freq:
            try:
                # Align to inferred frequency; if holidays cause holes, fall back to setting metadata only
                s = s.asfreq(freq)
            except ValueError:
                try:
                    s.index.freq = to_offset(freq)
                except Exception:
                    # If even that fails, keep as-is; SARIMAX will still work, just noisier
                    pass

    return s.astype(float)


def eval_symbol(
    path: Path, order: tuple[int, int, int] = (1, 1, 1), test_frac: float = 0.2
) -> dict[str, object]:
    """Fit SARIMAX(p,d,q) to train split, forecast test split, compute metrics."""
    sym = path.stem.upper()
    y = load_price_series(path)

    if len(y) < 50:
        raise ValueError(f"{sym}: not enough observations ({len(y)})")

    n_test = max(20, int(len(y) * test_frac))
    y_train, y_test = y.iloc[:-n_test], y.iloc[-n_test:]

    # Baseline SARIMAX without seasonality to avoid freq issues across symbols
    model = SARIMAX(
        y_train,
        order=order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
        warnings.filterwarnings("ignore", message="No frequency information")
        res = model.fit(disp=False)

    fc = res.forecast(steps=len(y_test))

    # Metrics
    e = fc - y_test
    rmse = float(np.sqrt(np.mean(np.square(e))))
    mae = float(np.mean(np.abs(e)))
    mape = float(np.mean(np.abs(e / y_test))) * 100.0
    aic, bic = float(res.aic), float(res.bic)

    return {
        "symbol": sym,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "model": f"SARIMAX{order}",
        "rmse": round(rmse, 6),
        "mae": round(mae, 6),
        "mape": round(mape, 4),
        "aic": round(aic, 3),
        "bic": round(bic, 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="W10 — ARIMAX/SARIMAX forecast evaluation")
    ap.add_argument("--data-glob", default="data/csv/*.csv", help="Glob for input price CSV files")
    ap.add_argument("--out", default="reports/wk10_forecast_eval.csv", help="Output CSV path")
    ap.add_argument("--order", default="1,1,1", help="ARIMA(p,d,q) e.g. 1,1,1")
    ap.add_argument(
        "--test-frac", type=float, default=0.2, help="Fraction of samples for test split"
    )
    args = ap.parse_args()

    try:
        p, d, q = (int(x) for x in args.order.split(","))
    except Exception as exc:  # noqa: BLE001
        print(f"--order must be like '1,1,1' (got {args.order}): {exc}", file=sys.stderr)
        sys.exit(2)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    paths = list(Path().glob(args.data_glob))
    if not paths:
        print(f"No files matched {args.data_glob}", file=sys.stderr)
        sys.exit(1)

    rows: list[dict[str, object]] = []
    for pth in paths:
        try:
            rows.append(eval_symbol(pth, order=(p, d, q), test_frac=args.test_frac))
        except Exception as ex:  # noqa: BLE001
            rows.append({"symbol": pth.stem.upper(), "error": str(ex)})

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path.as_posix()} with {len(df)} rows")


if __name__ == "__main__":
    main()

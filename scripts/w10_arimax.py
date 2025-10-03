"""
W10 — ARIMAX/SARIMAX forecast evaluator (hardened).

Reads price CSVs (via glob), fits a baseline SARIMAX(p,d,q) per symbol,
evaluates on a holdout, and writes metrics to reports/wk10_forecast_eval.csv.

Expected price CSV columns (flexible):
- Price: one of ["Adj Close","adj_close","Close","close","price","adjclose","adj_close_price"]
- Date : optional, one of ["Date","date","Timestamp","timestamp","dt"]

Optional exogenous CSV (for ARIMAX):
- Has a date-like column (as above) plus one or more numeric columns to use as regressors.
- Use --exog-csv to point to the file and --exog-cols with a comma-separated list of columns.
- We align exog to each symbol’s index; you can also lag exog via --exog-lag (periods).

Usage (PowerShell):
  python scripts/w10_arimax.py `
    --data-glob "data/csv/*.csv" `
    --out "reports/wk10_forecast_eval.csv" `
    --order "1,1,1" `
    --test-frac 0.25 `
    --exog-csv "data/factors/nifty_macro_extended.csv" `
    --exog-cols "mkt_ret,carry" `
    --exog-lag 1 `
    --maxiter 500 `
    --solver lbfgs `
    --retry-orders "0,1,1;2,1,1"
"""

from __future__ import annotations

import argparse
import sys
import warnings
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tools.sm_exceptions import ValueWarning as SMValueWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Quiet, harmless statsmodels date-frequency warnings
warnings.filterwarnings("ignore", category=SMValueWarning)

PRICE_CANDIDATES: Final[tuple[str, ...]] = (
    "adj close",
    "adj_close",
    "close",
    "price",
    "adjclose",
    "adj_close_price",
)
DATE_CANDIDATES: Final[tuple[str, ...]] = (
    "date",
    "timestamp",
    "dt",
)


# ----------------------------
# Small helpers
# ----------------------------
def _first_match(candidates: Iterable[str], haystack: Iterable[str]) -> str | None:
    """Case-insensitive first match; returns the ORIGINAL name from haystack if found."""
    lower_to_orig = {str(c).lower(): c for c in haystack}
    for want in candidates:
        if want in lower_to_orig:
            return lower_to_orig[want]
    return None


def _coerce_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Try to find a date-like column and set a DateTimeIndex sorted ascending."""
    date_col = _first_match(DATE_CANDIDATES, df.columns)
    if date_col:
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=False)
        df = df.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)
    return df


def _safe_mape(y_true: pd.Series, err: pd.Series) -> float:
    """MAPE that avoids division by zero and ignores NaN/inf."""
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = np.abs(err / y_true.replace(0, np.nan))
    return float(np.nanmean(pct) * 100.0)


def load_price_series(path: Path) -> pd.Series:
    """Load a price series from CSV; return float series indexed by date if present."""
    df = pd.read_csv(path)
    df = _coerce_datetime_index(df)

    price_col = _first_match(PRICE_CANDIDATES, df.columns)
    if not price_col:
        raise ValueError(f"No price-like column found in {path}")

    s = pd.to_numeric(df[price_col], errors="coerce").dropna()
    if isinstance(s.index, pd.DatetimeIndex) and not s.index.is_monotonic_increasing:
        s = s.sort_index()
    return s.astype(float)


def load_exog_frame(exog_csv: Path, exog_cols: Sequence[str]) -> pd.DataFrame:
    """
    Load exogenous variables CSV and return a frame with a DateTimeIndex (if present)
    containing ONLY the requested columns (case-insensitive name resolution).
    """
    exog_df = pd.read_csv(exog_csv)
    exog_df = _coerce_datetime_index(exog_df)

    # Resolve requested columns case-insensitively
    existing_lower = {c.lower(): c for c in exog_df.columns}
    resolved: list[str] = []
    missing: list[str] = []
    for c in exog_cols:
        key = c.strip().lower()
        if key in existing_lower:
            resolved.append(existing_lower[key])
        else:
            missing.append(c.strip())
    if missing:
        raise ValueError(f"exog columns not found in {exog_csv}: {', '.join(missing)}")

    out = exog_df[resolved].apply(pd.to_numeric, errors="coerce").dropna(how="all")
    return out


def _maybe_lag(df: pd.DataFrame, lag: int) -> pd.DataFrame:
    """Lag exogenous regressors by N periods to avoid look-ahead."""
    if lag <= 0:
        return df
    return df.shift(lag)


# ----------------------------
# Core
# ----------------------------
def eval_symbol(
    path: Path,
    order: tuple[int, int, int] = (1, 1, 1),
    test_frac: float = 0.2,
    exog: pd.DataFrame | None = None,
    exog_lag: int = 0,
    maxiter: int = 200,
    solver: str = "lbfgs",
    retry_orders: list[tuple[int, int, int]] | None = None,
) -> dict[str, Any]:
    """
    Fit SARIMAX(p,d,q) to train split, forecast test split, compute metrics.
    If exog provided → ARIMAX. Adds robust convergence handling.
    """
    sym = path.stem.upper()
    y = load_price_series(path)

    if len(y) < 50:
        raise ValueError(f"{sym}: not enough observations ({len(y)})")

    n_test = max(20, int(len(y) * test_frac))
    y_train, y_test = y.iloc[:-n_test], y.iloc[-n_test:]

    # Prepare exog aligned to y, with optional lag
    X = None
    X_train = None
    X_test = None
    exog_used = False

    if exog is not None:
        X = exog.reindex(y.index)
        if exog_lag:
            X = _maybe_lag(X, exog_lag)
        # If alignment creates leading NaNs, align y as well to common dates
        idx_common = X.dropna(how="all").index.intersection(y.index)
        y_aligned = y.reindex(idx_common).dropna()
        X_aligned = X.reindex(y_aligned.index).dropna(how="all")
        if len(y_aligned) >= 50:  # keep safety check
            y = y_aligned
            X = X_aligned
            n_test = max(20, int(len(y) * test_frac))
            y_train, y_test = y.iloc[:-n_test], y.iloc[-n_test:]
            X_train, X_test = X.iloc[:-n_test], X.iloc[-n_test:]
            exog_used = X_train.shape[1] > 0

    tried: list[tuple[int, int, int]] = [order] + (retry_orders or [])
    res = None
    fc = None
    fit_status = "ok"
    used_order: tuple[int, int, int] | None = None
    last_err: str | None = None

    for ord_ in tried:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                model = SARIMAX(
                    y_train,
                    exog=X_train,
                    order=ord_,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                # statsmodels fit: method is the optimizer; disp=False suppresses output
                res = model.fit(disp=False, maxiter=maxiter, method=solver)

            # Mark status if optimizer reports not converged
            if hasattr(res, "mle_retvals"):
                if not res.mle_retvals.get("converged", True):
                    fit_status = "warn_convergence"

            fc = res.forecast(steps=len(y_test), exog=X_test)
            used_order = ord_
            break

        except Exception as exc:  # noqa: BLE001
            last_err = f"{exc.__class__.__name__}: {exc}"
            fit_status = f"fail_{ord_}"
            continue

    if res is None or fc is None or used_order is None:
        return {"symbol": sym, "error": last_err or "fit_failed"}

    # Metrics
    e = (fc - y_test).astype(float)
    rmse = float(np.sqrt(np.mean(np.square(e))))
    mae = float(np.mean(np.abs(e)))
    mape = _safe_mape(y_test, e)
    aic, bic = float(res.aic), float(res.bic)

    return {
        "symbol": sym,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "model": f"{'ARIMAX' if exog_used else 'ARIMA'}{used_order}",
        "rmse": round(rmse, 6),
        "mae": round(mae, 6),
        "mape": round(mape, 4),
        "aic": round(aic, 3),
        "bic": round(bic, 3),
        "fit_status": fit_status,
        "exog_used": bool(exog_used),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="W10 — ARIMAX/SARIMAX forecast evaluation (hardened)")
    ap.add_argument("--data-glob", default="data/csv/*.csv", help="Glob for input price CSV files")
    ap.add_argument("--out", default="reports/wk10_forecast_eval.csv", help="Output CSV path")
    ap.add_argument("--order", default="1,1,1", help="ARIMA(p,d,q) e.g. 1,1,1")
    ap.add_argument(
        "--test-frac", type=float, default=0.2, help="Fraction of samples for test split"
    )

    # Exogenous controls (optional)
    ap.add_argument("--exog-csv", default=None, help="CSV containing exogenous variables")
    ap.add_argument(
        "--exog-cols",
        default=None,
        help="Comma-separated column names inside exog CSV (case-insensitive)",
    )
    ap.add_argument(
        "--exog-lag",
        type=int,
        default=0,
        help="Lag exogenous regressors by N periods (to avoid leakage).",
    )

    # Convergence hardening
    ap.add_argument("--maxiter", type=int, default=200, help="Max optimizer iterations")
    ap.add_argument(
        "--solver",
        default="lbfgs",
        choices=["lbfgs", "bfgs", "nm", "cg", "powell"],
        help="Optimizer for MLE",
    )
    ap.add_argument(
        "--retry-orders",
        default="0,1,1;2,1,1",
        help='Fallback orders, semicolon-separated, e.g. "0,1,1;2,1,1"',
    )

    args = ap.parse_args()

    # Parse order
    try:
        p, d, q = (int(x) for x in args.order.split(","))
    except Exception as exc:  # noqa: BLE001
        print(f"--order must be like '1,1,1' (got {args.order}): {exc}", file=sys.stderr)
        sys.exit(2)

    # Parse retry orders
    retry_orders: list[tuple[int, int, int]] = []
    if args.retry_orders:
        for chunk in args.retry_orders.split(";"):
            chunk = chunk.strip()
            if not chunk:
                continue
            parts = [int(x) for x in chunk.split(",")]
            if len(parts) == 3:
                retry_orders.append((parts[0], parts[1], parts[2]))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare optional exog
    exog_frame: pd.DataFrame | None = None
    if args.exog_csv and args.exog_cols:
        exog_cols = [c.strip() for c in str(args.exog_cols).split(",") if c.strip()]
        exog_frame = load_exog_frame(Path(args.exog_csv), exog_cols)
        exog_frame = _maybe_lag(exog_frame, args.exog_lag)

    paths = list(Path().glob(args.data_glob))
    if not paths:
        print(f"No files matched {args.data_glob}", file=sys.stderr)
        sys.exit(1)

    rows: list[dict[str, Any]] = []
    for pth in paths:
        try:
            rows.append(
                eval_symbol(
                    pth,
                    order=(p, d, q),
                    test_frac=args.test_frac,
                    exog=exog_frame,
                    exog_lag=args.exog_lag,
                    maxiter=args.maxiter,
                    solver=args.solver,
                    retry_orders=retry_orders,
                )
            )
        except Exception as ex:  # noqa: BLE001
            rows.append({"symbol": pth.stem.upper(), "error": str(ex)})

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path.as_posix()} with {len(df)} rows")


if __name__ == "__main__":
    main()

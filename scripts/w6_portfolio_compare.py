# scripts/w6_portfolio_compare.py
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_returns(glob: str) -> pd.DataFrame:
    """
    Read all CSVs under `glob`, align by date, and return a wide dataframe of daily returns.
    Assumes each CSV has at least columns: date, close (or adj_close).
    """
    rets: list[pd.Series] = []
    for p in sorted(Path().glob(glob)):
        if p.suffix.lower() != ".csv":
            continue
        try:
            df = pd.read_csv(p)
        except Exception:
            continue

        sym = p.stem.upper()
        # pick a price column
        price_col = None
        for c in ("adj_close", "close", "Close", "Adj Close"):
            if c in df.columns:
                price_col = c
                break
        if price_col is None:
            continue

        if "date" not in df.columns:
            # try common alternatives
            for c in ("Date", "timestamp"):
                if c in df.columns:
                    df = df.rename(columns={c: "date"})
                    break

        if "date" not in df.columns:
            continue

        # compute pct change
        df["date"] = pd.to_datetime(df["date"])
        s = df.sort_values("date")[price_col].astype(float).pct_change().rename(sym)
        rets.append(s)

    if not rets:
        raise RuntimeError(f"No usable CSVs found for pattern: {glob}")

    wide = pd.concat(rets, axis=1)
    wide = wide.dropna(how="any")  # align on common dates
    return wide


def min_var_weights(cov: np.ndarray) -> np.ndarray:
    """
    Unconstrained minimum-variance weights: w ∝ Σ^{-1} 1, normalized to sum=1.
    """
    n = cov.shape[0]
    ones = np.ones(n)
    try:
        inv = np.linalg.pinv(cov)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(cov + 1e-8 * np.eye(n))
    raw = inv @ ones
    if raw.sum() <= 0:
        # Fallback: equal weight
        return np.ones(n) / n
    w = raw / raw.sum()
    return w


def diag_shrinkage(cov: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    """
    Simple diagonal shrinkage towards the diagonal of Σ:
      Σ_shrink = (1-α) Σ + α * diag(diag(Σ))
    α in [0,1]. This is a lightweight Ledoit–Wolf-style baseline (not exact LW).
    """
    alpha = float(np.clip(alpha, 0.0, 1.0))
    diag = np.diag(np.diag(cov))
    return (1.0 - alpha) * cov + alpha * diag


def compute_allocations(rets: pd.DataFrame, shrink_alpha: float = 0.2) -> pd.DataFrame:
    """
    Build a comparison table with EW and shrinkage MV weights.
    We clip negatives at 0 and renormalize to keep things practical for a baseline.
    """
    symbols = list(rets.columns)
    n = len(symbols)

    # Equal-weight
    w_ew = np.ones(n) / n

    # Sample covariance & shrinkage
    cov = np.cov(rets.values, rowvar=False)
    cov_s = diag_shrinkage(cov, alpha=shrink_alpha)

    # Unconstrained MV → clip negatives to 0 → renormalize
    w_mv = min_var_weights(cov_s)
    w_mv = np.where(w_mv < 0, 0.0, w_mv)
    s = w_mv.sum()
    if s <= 0:
        w_mv = w_ew.copy()
    else:
        w_mv = w_mv / s

    out = pd.DataFrame(
        {
            "symbol": symbols,
            "weight_ew": w_ew,
            "weight_mv_shrink": w_mv,
            # Capacity/sector flags are placeholders for now (true).
            # You can wire real ADV/sector checks later.
            "adv_cap_ok": True,
            "sector_cap_ok": True,
        }
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="W6: Portfolio compare (EW vs shrinkage MV)")
    ap.add_argument("--data-glob", default="data/csv/*.csv", help="Input CSV glob")
    ap.add_argument("--out", default="reports/wk6_portfolio_compare.csv", help="Output CSV path")
    ap.add_argument(
        "--shrink-alpha", type=float, default=0.2, help="Diagonal shrinkage intensity [0,1]"
    )
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rets = load_returns(args.data_glob)
    table = compute_allocations(rets, shrink_alpha=args.shrink_alpha)
    table.to_csv(out_path, index=False)
    print(f"Wrote {out_path} with {len(table)} rows")


if __name__ == "__main__":
    main()

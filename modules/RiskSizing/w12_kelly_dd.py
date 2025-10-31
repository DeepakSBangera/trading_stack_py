# scripts/w12_kelly_dd.py
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _synthetic_equity(
    n: int = 252, mu: float = 0.12, vol: float = 0.20
) -> pd.DataFrame:
    """Build a tiny synthetic daily equity curve if no input is provided."""
    rng = np.random.default_rng(42)
    dt = 1 / 252
    rets = rng.normal(loc=mu * dt, scale=vol * np.sqrt(dt), size=n)
    equity = 1.0 * np.cumprod(1 + rets)
    dates = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame({"date": dates, "equity": equity})


def _rolling_dd(equity: pd.Series) -> pd.Series:
    """Rolling drawdown (peak-to-trough), as a positive fraction (e.g., 0.08 for -8%)."""
    roll_max = equity.cummax()
    dd = (roll_max - equity) / roll_max
    return dd.fillna(0.0)


def _kelly_from_stats(mu: float, sigma: float) -> float:
    """Toy Kelly: f* = mu / sigma^2  (clipped into [0,1])."""
    if sigma <= 0:
        return 0.0
    k = mu / (sigma * sigma)
    return float(np.clip(k, 0.0, 1.0))


def _throttle_from_dd(dd: float, soft: float = 0.05, hard: float = 0.10) -> float:
    """
    Throttle in [0,1] as a simple piecewise:
      - dd <= soft  -> 1.0
      - soft<dd<hard-> linearly drop to 0.5
      - dd >= hard  -> 0.5
    """
    if dd <= soft:
        return 1.0
    if dd >= hard:
        return 0.5
    # linear: soft -> 1.0  down to  hard -> 0.5
    span = hard - soft
    frac = (dd - soft) / span
    return float(1.0 - 0.5 * frac)


def run(out_path: Path, equity_csv: str | None, target_vol: float) -> None:
    if equity_csv and Path(equity_csv).exists():
        df = pd.read_csv(equity_csv)
        # try to find date/equity columns
        date_col = next(
            (c for c in df.columns if str(c).lower() in ("date", "timestamp")), None
        )
        eq_col = next(
            (c for c in df.columns if str(c).lower() in ("equity", "nav", "pnl")), None
        )
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.dropna(subset=[date_col]).sort_values(date_col)
        else:
            # synthesize a date index if needed
            df["date"] = pd.bdate_range("2024-01-01", periods=len(df))
            date_col = "date"
        if not eq_col:
            raise SystemExit("Could not find equity/NAV column in input CSV.")
        df = df[[date_col, eq_col]].rename(columns={date_col: "date", eq_col: "equity"})
    else:
        df = _synthetic_equity()

    # daily returns
    df["ret"] = df["equity"].pct_change().fillna(0.0)

    # simple rolling stats (60d)
    win = 60
    mu = df["ret"].rolling(win).mean() * 252.0
    sigma = df["ret"].rolling(win).std() * np.sqrt(252.0)

    # Kelly & throttle
    df["kelly_fraction"] = [
        _kelly_from_stats(m if np.isfinite(m) else 0.0, s if np.isfinite(s) else 0.0)
        for m, s in zip(mu, sigma, strict=False)
    ]
    dd = _rolling_dd(df["equity"])
    df["dd_throttle"] = [_throttle_from_dd(float(x)) for x in dd]

    # position scale: cap Kelly at 1, apply throttle, scale to target_vol if you want
    df["target_vol"] = float(target_vol)
    df["position_scale"] = (df["kelly_fraction"].clip(0.0, 1.0)) * df["dd_throttle"]

    out = df[
        ["date", "kelly_fraction", "target_vol", "dd_throttle", "position_scale"]
    ].copy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path.as_posix()} with {len(out)} rows")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="W12 — Kelly fraction with drawdown throttle"
    )
    ap.add_argument("--out", default="reports/wk12_kelly_dd.csv")
    ap.add_argument(
        "--equity-csv", default=None, help="Optional equity/NAV time series CSV"
    )
    ap.add_argument(
        "--target-vol", type=float, default=0.20, help="Target vol used for reporting"
    )
    args = ap.parse_args()
    run(Path(args.out), args.equity_csv, args.target_vol)


if __name__ == "__main__":
    main()

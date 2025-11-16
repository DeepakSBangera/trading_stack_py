import argparse
import pathlib

import numpy as np
import pandas as pd


def find_latest_equity(reports: pathlib.Path) -> pathlib.Path:
    cands = [
        p
        for p in reports.glob("portfolioV2_*.parquet")
        if not (
            p.name.endswith("_weights.parquet")
            or p.name.endswith("_trades.parquet")
            or p.name.endswith("_tearsheet.parquet")
        )
    ]
    if not cands:
        raise SystemExit(f"No equity parquet found in {reports}")
    return max(cands, key=lambda p: p.stat().st_mtime)


def load_equity(path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    if "date" not in df.columns:
        raise SystemExit("equity-like file missing 'date'")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    # pick a numeric equity column
    prefer = ["equity", "nav", "portfolio", "value", "equity_curve", "equity_nav"]
    eq_col = next((c for c in prefer if c in df.columns), None)
    if eq_col is None:
        num_cols = [c for c in df.columns if c != "date" and pd.api.types.is_numeric_dtype(df[c])]
        if not num_cols:
            raise SystemExit("No numeric equity-like column found.")
        eq_col = num_cols[-1]
    df["equity"] = pd.to_numeric(df[eq_col], errors="coerce")
    df["equity"] = df["equity"].replace([np.inf, -np.inf], np.nan).interpolate().bfill().ffill()
    return df[["date", "equity"]]


def ann_vol(returns: pd.Series, win: int) -> pd.Series:
    # rolling std with ddof=0, annualized
    return returns.rolling(win, min_periods=win).std(ddof=0) * np.sqrt(252)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--equity", default="")  # optional override
    ap.add_argument("--vol-wins", default="20,63,126")  # rolling windows (days)
    ap.add_argument("--target-vol", type=float, default=0.0)  # 0 = compute, but don't size
    args = ap.parse_args()

    reports = pathlib.Path(args.reports)
    if not reports.exists():
        raise SystemExit(f"reports folder not found: {reports}")

    eq_path = pathlib.Path(args.equity) if args.equity else find_latest_equity(reports)
    eqdf = load_equity(eq_path)

    # daily returns
    ret = eqdf["equity"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # rolling vols
    wins = [int(x) for x in str(args.vol_wins).split(",") if str(x).strip()]
    out = eqdf.copy()
    out["ret"] = ret
    for w in wins:
        out[f"vol{w}"] = ann_vol(ret, w)

    # a simple ATR-like stop proxy using rolling drawdown depth
    peak = out["equity"].cummax()
    dd = out["equity"] / peak - 1.0
    out["drawdown"] = dd
    out["maxdd_to_date"] = dd.cummin()

    # if target vol requested, compute naive sizing multiplier (cap 0..1.5 for display)
    if args.target_vol > 0:
        # use vol63 if present, else first window
        vcol = "vol63" if "vol63" in out.columns else f"vol{wins[0]}"
        vc = out[vcol].replace(0.0, np.nan)
        mult = (args.target_vol / vc).clip(upper=1.5)
        out["target_weight"] = mult.fillna(0.0)
    else:
        out["target_weight"] = 0.0

    out_path = reports / "wk4_voltarget_stops.parquet"
    out.to_parquet(out_path, index=False, compression="snappy")
    print(f"✓ Wrote {out_path}")
    print(f"Source equity: {eq_path}")


if __name__ == "__main__":
    main()

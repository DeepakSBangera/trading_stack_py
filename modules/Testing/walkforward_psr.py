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


def max_drawdown(series: pd.Series) -> float:
    peak = series.cummax()
    dd = series / peak - 1.0
    return float(dd.min())


def fold_metrics(df: pd.DataFrame) -> dict:
    # normalize equity to 1 at start for the fold
    eq = df["equity"]
    ret = eq.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    n = len(ret) if len(ret) > 0 else 1
    cagr = (1.0 + ret).prod() ** (252 / max(n, 1)) - 1.0
    vol = ret.std(ddof=0) * np.sqrt(252)
    sharpe = (ret.mean() * 252) / (ret.std(ddof=0) + 1e-12)
    mdd = max_drawdown(eq)
    return {
        "start_date": df["date"].iloc[0].date().isoformat(),
        "end_date": df["date"].iloc[-1].date().isoformat(),
        "points": int(len(df)),
        "CAGR": float(cagr),
        "Sharpe": float(sharpe),
        "MaxDD": float(mdd),
        "VolAnn": float(vol),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--equity", default="")  # optional explicit equity path
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    reports = pathlib.Path(args.reports)
    eq_path = pathlib.Path(args.equity) if args.equity else find_latest_equity(reports)
    eqdf = load_equity(eq_path)

    # split sequentially into N folds with near-equal sizes
    idxs = np.array_split(np.arange(len(eqdf)), max(1, args.folds))
    rows = []
    for i, ix in enumerate(idxs, start=1):
        if len(ix) < 2:  # skip empty/tiny folds
            continue
        sub = eqdf.iloc[ix]
        m = fold_metrics(sub)
        m["fold"] = i
        rows.append(m)

    out = pd.DataFrame(
        rows,
        columns=[
            "fold",
            "start_date",
            "end_date",
            "points",
            "CAGR",
            "Sharpe",
            "MaxDD",
            "VolAnn",
        ],
    )
    out_path = reports / "wk5_walkforward.parquet"
    out.to_parquet(out_path, index=False, compression="snappy")
    print(f"✓ Wrote {out_path}")
    print(f"Source equity: {eq_path}")


if __name__ == "__main__":
    main()

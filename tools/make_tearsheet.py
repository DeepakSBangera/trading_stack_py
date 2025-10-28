import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def daily_returns(eq: pd.Series) -> pd.Series:
    r = eq.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return r


def max_drawdown(eq: pd.Series):
    cummax = eq.cummax()
    dd = eq / cummax - 1.0
    return dd.min(), dd


def main():
    ap = argparse.ArgumentParser()
    # accepts CSV or Parquet, keeping arg name for backward compat
    ap.add_argument("--equity-csv", required=True)
    args = ap.parse_args()

    def _load(path):
        return (
            pd.read_parquet(path)
            if path.lower().endswith(".parquet")
            else pd.read_csv(path)
        )

    eqdf = _load(args.equity_csv)
    if "date" not in eqdf.columns:
        raise SystemExit("equity file missing 'date'")
    eqdf["date"] = pd.to_datetime(eqdf["date"], errors="coerce")
    eqdf = eqdf.dropna(subset=["date"]).sort_values("date")

    # pick equity-like column
    eq_col = next(
        (
            c
            for c in (
                "equity",
                "nav",
                "portfolio",
                "value",
                "equity_curve",
                "equity_nav",
            )
            if c in eqdf.columns
        ),
        None,
    )
    if eq_col is None:
        num_cols = [
            c
            for c in eqdf.columns
            if c != "date" and pd.api.types.is_numeric_dtype(eqdf[c])
        ]
        if not num_cols:
            raise SystemExit("No numeric equity-like column found.")
        eq_col = num_cols[-1]

    eq = (
        pd.to_numeric(eqdf[eq_col], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .interpolate()
        .bfill()
        .ffill()
    )
    ret = daily_returns(eq)

    # summary stats
    n = max(1, len(ret))
    ann = (1.0 + ret).prod() ** (252 / n) - 1.0
    vol = ret.std(ddof=0) * np.sqrt(252)
    mdd, ddser = max_drawdown(eq)
    sharpe = (ret.mean() * 252) / (ret.std(ddof=0) + 1e-12)
    calmar = (ann / abs(mdd)) if mdd != 0 else np.nan

    base = os.path.splitext(args.equity_csv)[0].replace("_tearsheet", "")
    out_png = f"{base}_tearsheet.png"
    out_pq = f"{base}_tearsheet.parquet"

    # Plot (simple)
    plt.figure(figsize=(10, 6))
    plt.plot(eqdf["date"], eq.values)
    plt.title("Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

    # Build a Parquet-friendly table: numeric in value_num, strings in value_str
    rows = [
        {"metric": "CAGR", "value_num": float(ann), "value_str": None},
        {"metric": "Sharpe", "value_num": float(sharpe), "value_str": None},
        {"metric": "MaxDD", "value_num": float(mdd), "value_str": None},
        {"metric": "Calmar", "value_num": float(calmar), "value_str": None},
        {
            "metric": "first_date",
            "value_num": None,
            "value_str": eqdf["date"].min().date().isoformat(),
        },
        {
            "metric": "last_date",
            "value_num": None,
            "value_str": eqdf["date"].max().date().isoformat(),
        },
        {"metric": "points", "value_num": float(len(eqdf)), "value_str": None},
        {"metric": "vol_ann", "value_num": float(vol), "value_str": None},
    ]
    ts = pd.DataFrame(rows, columns=["metric", "value_num", "value_str"])
    ts.to_parquet(out_pq, index=False, compression="snappy")

    print(f"✓ Wrote {out_pq}")
    print(f"✓ Wrote {out_png}")


if __name__ == "__main__":
    main()

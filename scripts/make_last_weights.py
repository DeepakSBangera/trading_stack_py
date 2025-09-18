from pathlib import Path

import pandas as pd

base = Path("reports/backtests")
runs = sorted(
    [d for d in base.iterdir() if d.is_dir()], key=lambda d: d.stat().st_mtime, reverse=True
)
if not runs:
    raise SystemExit("No backtest runs found in reports/backtests")

# Prefer a run that already has monthly_weights.csv; otherwise rebuild from px.csv.
folder = next((r for r in runs if (r / "monthly_weights.csv").exists()), None)

if folder:
    mw = pd.read_csv(folder / "monthly_weights.csv", index_col=0, parse_dates=True)
    last = mw.tail(1).T
    last.columns = ["weight"]
    out = folder / "last_weights.csv"
    last.to_csv(out)
    print("Wrote:", out)
else:
    # Fallback: use latest run, compute last month’s weights using R12-1 on px.csv
    folder = runs[0]
    px_path = folder / "px.csv"
    if not px_path.exists():
        raise SystemExit(f"No monthly_weights.csv in any run, and {px_path} not found to rebuild.")

    px = pd.read_csv(px_path, index_col=0, parse_dates=True)
    # Month-end prices
    px_m = px.resample("ME").last()

    # Simple R12-1 momentum score: 12m return minus 1m return (excluding the most recent month)
    score = (px_m / px_m.shift(12) - 1) - (px_m / px_m.shift(1) - 1)

    last_score = score.iloc[-1].dropna().sort_values(ascending=False)
    TOP_N = 5
    chosen = last_score.head(TOP_N).index

    w = pd.Series(0.0, index=px.columns)
    if len(chosen) > 0:
        w.loc[chosen] = 1.0 / len(chosen)
    w.name = "weight"

    out = folder / "last_weights.csv"
    w.to_csv(out)
    print("Rebuilt from px.csv and wrote:", out)

print("Run folder:", folder)

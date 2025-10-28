from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

TOP_N = 5
base = Path("reports/backtests")
base.mkdir(parents=True, exist_ok=True)
runs = sorted(
    [d for d in base.iterdir() if d.is_dir()],
    key=lambda d: d.stat().st_mtime,
    reverse=True,
)

# 1) Prefer monthly_weights.csv if available in any run (newest first)
for r in runs:
    mw = r / "monthly_weights.csv"
    if mw.exists():
        df = pd.read_csv(mw, index_col=0, parse_dates=True)
        last = df.tail(1).T
        last.columns = ["weight"]
        out = r / "last_weights.csv"
        last.to_csv(out)
        print("Wrote:", out)
        raise SystemExit(0)

# 2) Else fallback to px.csv if available (newest first)
for r in runs:
    pxp = r / "px.csv"
    if pxp.exists():
        px = pd.read_csv(pxp, index_col=0, parse_dates=True)
        px_m = px.resample("ME").last()
        score = (px_m / px_m.shift(12) - 1) - (px_m / px_m.shift(1) - 1)
        last_score = score.iloc[-1].dropna().sort_values(ascending=False)
        chosen = last_score.head(TOP_N).index
        w = pd.Series(0.0, index=px.columns)
        if len(chosen):
            w.loc[chosen] = 1.0 / len(chosen)
        out = r / "last_weights.csv"
        w.to_csv(out, header=["weight"])
        print("Rebuilt from:", pxp)
        print("Wrote:", out)
        raise SystemExit(0)

# 3) Else compute straight from raw parquet files
prices_dir = Path("data/prices")
parqs = sorted(prices_dir.glob("*.parquet"))

if parqs:
    PRICE_CANDIDATES = ["adj close", "Adj Close", "close", "Close"]

    def load_parquet_dir(path: Path) -> pd.DataFrame:
        frames = []
        for p in sorted(path.glob("*.parquet")):
            try:
                df = pd.read_parquet(p)
                if not isinstance(df.index, pd.DatetimeIndex):
                    if "date" in df.columns:
                        df = df.set_index(
                            pd.to_datetime(df["date"], errors="coerce")
                        ).drop(columns=["date"])
                    else:
                        try:
                            df.index = pd.to_datetime(df.index, errors="coerce")
                        except Exception:
                            pass
                price_col = None
                for c in PRICE_CANDIDATES:
                    if c in df.columns:
                        price_col = c
                        break
                if price_col is None and df.shape[1] == 1:
                    price_col = df.columns[0]
                if price_col is None or not isinstance(df.index, pd.DatetimeIndex):
                    continue
                s = pd.to_numeric(df[price_col], errors="coerce").rename(p.stem)
                s = s[~s.index.isna()].sort_index()
                if len(s):
                    frames.append(s)
            except Exception:
                pass
        if not frames:
            raise SystemExit("Could not build from raw parquet files.")
        return pd.concat(frames, axis=1).sort_index()

    px = load_parquet_dir(prices_dir)
    px_m = px.resample("ME").last()
    score = (px_m / px_m.shift(12) - 1) - (px_m / px_m.shift(1) - 1)
    last_score = score.iloc[-1].dropna().sort_values(ascending=False)
    chosen = last_score.head(TOP_N).index
    w = pd.Series(0.0, index=px.columns)
    if len(chosen):
        w.loc[chosen] = 1.0 / len(chosen)

    adhoc = base / ("adhoc_" + datetime.now().strftime("%Y-%m-%d_%H%M"))
    adhoc.mkdir(parents=True, exist_ok=True)
    out = adhoc / "last_weights.csv"
    w.to_csv(out, header=["weight"])
    print("Built from raw data/prices parquet files.")
    print("Wrote:", out)
    print("Run folder:", adhoc)
else:
    raise SystemExit(
        "No monthly_weights.csv or px.csv found in runs, and no parquet files in data/prices."
    )

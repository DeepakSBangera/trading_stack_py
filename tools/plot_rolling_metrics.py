from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

IN_PARQUET = Path("reports/rolling_metrics.parquet")
OUT_PNG = Path("reports/rolling_metrics.png")


def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure tz-naive DatetimeIndex named 'date'
    if df.index.name != "date":
        if "date" in df.columns:
            idx = pd.to_datetime(df["date"], utc=True, errors="coerce")
        else:
            idx = pd.to_datetime(df.index, utc=True, errors="coerce")
        try:
            idx = idx.tz_convert(None)
        except Exception:
            pass
        try:
            idx = idx.tz_localize(None)
        except Exception:
            pass
        df = df.copy()
        df.index = idx
        df.index.name = "date"
        if "date" in df.columns:
            df = df.drop(columns=["date"])
    return df.sort_index()


def main():
    if not IN_PARQUET.exists():
        print(f"[ERROR] Missing input: {IN_PARQUET}")
        sys.exit(2)

    df = pd.read_parquet(IN_PARQUET)
    df = _normalize_index(df)

    # Expected columns
    cols = ["rolling_vol", "rolling_sharpe", "rolling_sortino", "rolling_mdd", "regime"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"[ERROR] Missing columns in {IN_PARQUET.name}: {missing}")
        sys.exit(3)

    # Make figure: 5 stacked plots
    fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(14, 10), sharex=True)
    ax_vol, ax_sh, ax_so, ax_dd, ax_rg = axes

    ax_vol.plot(df.index, df["rolling_vol"])
    ax_vol.set_ylabel("Vol (ann)")
    ax_vol.grid(True, alpha=0.3)

    ax_sh.plot(df.index, df["rolling_sharpe"])
    ax_sh.set_ylabel("Sharpe")
    ax_sh.grid(True, alpha=0.3)

    ax_so.plot(df.index, df["rolling_sortino"])
    ax_so.set_ylabel("Sortino")
    ax_so.grid(True, alpha=0.3)

    ax_dd.plot(df.index, df["rolling_mdd"])
    ax_dd.set_ylabel("MaxDD")
    ax_dd.grid(True, alpha=0.3)

    ax_rg.step(df.index, df["regime"], where="post")
    ax_rg.set_ylabel("Regime")
    ax_rg.set_yticks([-1, 0, 1])
    ax_rg.grid(True, alpha=0.3)

    fig.suptitle("Rolling Metrics (Session 2)", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150)
    plt.close(fig)

    print(f"[OK] Wrote: {OUT_PNG}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(r"F:\Projects\trading_stack_py")
REPORTS = ROOT / "reports"
OUT = REPORTS / "factor_exposure_weekly.csv"

SECTOR_MAP = {
    "HDFCBANK.NS": "Financials",
    "RELIANCE.NS": "Energy",
    "TCS.NS": "IT",
    "INFY.NS": "IT",
    "AXISBANK.NS": "Financials",
    "ICICIBANK.NS": "Financials",
    "SBIN.NS": "Financials",
    "BHARTIARTL.NS": "Telecom",
    "ITC.NS": "Staples",
    "LT.NS": "Industrials",
    "KOTAKBANK.NS": "Financials",
    "BAJFINANCE.NS": "Financials",
    "HCLTECH.NS": "IT",
    "MARUTI.NS": "Discretionary",
    "SUNPHARMA.NS": "Healthcare",
    "TITAN.NS": "Discretionary",
    "ULTRACEMCO.NS": "Materials",
    "ONGC.NS": "Energy",
    "NESTLEIND.NS": "Staples",
    "ASIANPAINT.NS": "Materials",
}


def open_win(p: Path):
    if sys.platform.startswith("win") and p.exists():
        os.startfile(p)


def main():
    pq = REPORTS / "positions_daily.parquet"
    if not pq.exists():
        raise SystemExit(
            "Missing reports\\positions_daily.parquet — run W3 bootstrap first."
        )

    pos = pd.read_parquet(pq)
    pos["date"] = pd.to_datetime(pos["date"])
    pos["sector"] = pos["ticker"].map(SECTOR_MAP).fillna("Other")
    pos["week"] = pos["date"] - pd.to_timedelta(pos["date"].dt.weekday, unit="D")

    # portfolio weight per sector per week
    wk = (
        pos.groupby(["week", "sector"])["weight"]
        .sum()
        .reset_index()
        .pivot(index="week", columns="sector", values="weight")
        .fillna(0.0)
        .sort_index()
    )
    wk["gross_abs"] = (
        pos.assign(absw=pos["weight"].abs())
        .groupby("week")["absw"]
        .sum()
        .reindex(wk.index)
        .fillna(0.0)
    )
    wk.to_csv(OUT, index=True)

    print(
        json.dumps(
            {
                "out_csv": str(OUT),
                "weeks": wk.shape[0],
                "sectors": [c for c in wk.columns if c != "gross_abs"],
            },
            indent=2,
        )
    )

    open_win(OUT)


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"F:\Projects\trading_stack_py")
REPORTS = ROOT / "reports"
OUT = REPORTS / "events_calendar.csv"

TICKERS = [
    "HDFCBANK.NS",
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "AXISBANK.NS",
    "ICICIBANK.NS",
    "SBIN.NS",
    "BHARTIARTL.NS",
    "ITC.NS",
    "LT.NS",
    "KOTAKBANK.NS",
    "BAJFINANCE.NS",
    "HCLTECH.NS",
    "MARUTI.NS",
    "SUNPHARMA.NS",
    "TITAN.NS",
    "ULTRACEMCO.NS",
    "ONGC.NS",
    "NESTLEIND.NS",
    "ASIANPAINT.NS",
]


def open_win(p: Path):
    if sys.platform.startswith("win") and p.exists():
        os.startfile(p)


def to_iso_date(x) -> str:
    """Return YYYY-MM-DD from pandas/NumPy/py datetime types."""
    return pd.Timestamp(x).date().isoformat()


def main():
    pos_pq = REPORTS / "positions_daily.parquet"
    if not pos_pq.exists():
        raise SystemExit(
            "Missing reports\\positions_daily.parquet — run W3 bootstrap first."
        )
    pos = pd.read_parquet(pos_pq).sort_values("date")
    dates = pd.to_datetime(pos["date"].unique())
    if len(dates) == 0:
        raise SystemExit("No dates found in positions_daily.parquet")

    start, end = dates.min(), dates.max()

    rng = np.random.default_rng(101)
    # choose earnings dates strictly inside window if possible
    pick_pool = dates[2:-2] if len(dates) > 4 else dates
    earnings_rows = []
    for t in TICKERS:
        d = rng.choice(pick_pool)
        earnings_rows.append(
            {
                "date": to_iso_date(d),
                "ticker": t,
                "event_type": "earnings",
                "session": "post",
                "description": "Synthetic Q results",
            }
        )

    # market holidays within range (use business days if available)
    biz = pd.bdate_range(start, end)
    if len(biz) >= 10:
        holis = [biz[5], biz[min(12, len(biz) - 1)]]
    else:
        holis = [start, end]
    holiday_rows = [
        {
            "date": to_iso_date(d),
            "ticker": "",
            "event_type": "holiday",
            "session": "full-day",
            "description": "Market Holiday",
        }
        for d in holis
    ]

    cal = pd.DataFrame(
        earnings_rows + holiday_rows,
        columns=["date", "ticker", "event_type", "session", "description"],
    )
    REPORTS.mkdir(parents=True, exist_ok=True)
    cal.to_csv(OUT, index=False)

    print(json.dumps({"out_csv": str(OUT), "rows": int(cal.shape[0])}, indent=2))
    open_win(OUT)


if __name__ == "__main__":
    main()

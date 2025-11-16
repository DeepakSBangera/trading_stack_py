from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"F:\Projects\trading_stack_py")
REPORTS = ROOT / "reports"
CONFIG = ROOT / "config" / "capacity_policy.yaml"

# 20 large/liquid names to reduce per-name size
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
DATES = pd.date_range("2025-03-31", periods=20, freq="B")


def open_win(p: Path):
    if sys.platform.startswith("win") and p.exists():
        os.startfile(p)


def make_positions():
    rng = np.random.default_rng(123)
    rows = []
    for d in DATES:
        port_val = 10_000_000.0 + rng.normal(0, 30_000)  # ~₹1Cr
        base_w = 1.0 / len(TICKERS)  # ≈5%
        for t in TICKERS:
            w = base_w + rng.normal(0, 0.005)  # ±0.5% wiggle
            w = float(np.clip(w, 0.0, 0.12))
            rows.append((d, t, w, max(0.0, w * port_val), "L1", port_val))
    pd.DataFrame(
        rows,
        columns=[
            "date",
            "ticker",
            "weight",
            "position_value",
            "list_tier",
            "port_value",
        ],
    ).to_parquet(REPORTS / "positions_daily.parquet", index=False)


def make_adv():
    rng = np.random.default_rng(77)
    rows = []
    for d in DATES:
        for t in TICKERS:
            adv_val = rng.uniform(10_000_000, 40_000_000)  # ₹1–4Cr
            rows.append((d, t, adv_val))
    pd.DataFrame(rows, columns=["date", "ticker", "adv_value"]).to_parquet(REPORTS / "adv_value.parquet", index=False)


def make_orders():
    pos = pd.read_parquet(REPORTS / "positions_daily.parquet").sort_values(["ticker", "date"])
    pos["prev_val"] = pos.groupby("ticker")["position_value"].shift(1).fillna(0.0)
    pos["order_value"] = (pos["position_value"] - pos["prev_val"]).abs()
    pos[["date", "ticker", "order_value", "list_tier", "port_value"]].to_parquet(
        REPORTS / "orders_daily.parquet", index=False
    )


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    if not CONFIG.exists():
        raise SystemExit("Missing config\\capacity_policy.yaml — complete Step 1 first.")
    make_positions()
    make_adv()
    make_orders()
    print("Reseeded bootstrap OK →", REPORTS)
    open_win(REPORTS)


if __name__ == "__main__":
    main()

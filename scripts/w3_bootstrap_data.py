from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"F:\Projects\trading_stack_py")  # adjust if your repo path differs
REPORTS = ROOT / "reports"
CONFIG = ROOT / "config" / "capacity_policy.yaml"

TICKERS = ["HDFCBANK.NS", "RELIANCE.NS", "TCS.NS", "INFY.NS", "AXISBANK.NS"]
DATES = pd.date_range("2025-03-31", periods=20, freq="B")


def make_positions():
    rng = np.random.default_rng(42)
    rows = []
    for d in DATES:
        port_val = 10_000_000.0 + rng.normal(0, 50_000)
        for t in TICKERS:
            base_w = 1.0 / len(TICKERS)
            w = max(-0.05, min(0.25, base_w + rng.normal(0, 0.02)))
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
    rng = np.random.default_rng(7)
    rows = [(d, t, rng.uniform(5_000_000, 30_000_000)) for d in DATES for t in TICKERS]
    pd.DataFrame(rows, columns=["date", "ticker", "adv_value"]).to_parquet(
        REPORTS / "adv_value.parquet", index=False
    )


def make_orders():
    pos = pd.read_parquet(REPORTS / "positions_daily.parquet").sort_values(
        ["ticker", "date"]
    )
    pos["prev_val"] = pos.groupby("ticker")["position_value"].shift(1).fillna(0.0)
    pos["order_value"] = (pos["position_value"] - pos["prev_val"]).abs()
    pos[["date", "ticker", "order_value", "list_tier", "port_value"]].to_parquet(
        REPORTS / "orders_daily.parquet", index=False
    )


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    if not CONFIG.exists():
        raise SystemExit("Missing config\\capacity_policy.yaml — run Step 1 first.")
    if not (REPORTS / "positions_daily.parquet").exists():
        make_positions()
    if not (REPORTS / "adv_value.parquet").exists():
        make_adv()
    if not (REPORTS / "orders_daily.parquet").exists():
        make_orders()
    print("Bootstrap OK →", REPORTS)
    if sys.platform.startswith("win"):
        os.startfile(REPORTS)  # open the reports folder


if __name__ == "__main__":
    main()

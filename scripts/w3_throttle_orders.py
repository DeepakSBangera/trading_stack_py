from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"F:\Projects\trading_stack_py")
REPORTS = ROOT / "reports"
CONFIG = ROOT / "config" / "capacity_policy.yaml"


def open_win(p: Path):
    if sys.platform.startswith("win") and p.exists():
        os.startfile(p)


def load_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main():
    orders_p = REPORTS / "orders_daily.parquet"
    if not orders_p.exists():
        raise SystemExit(f"Missing {orders_p}. Run your bootstrap first.")

    pol = load_yaml(CONFIG)
    bands = pol.get("turnover_bands_pct_per_day", {"L1": 0.8, "L2": 1.2, "L3": 1.8})

    od = pd.read_parquet(orders_p).copy()
    # Expect: date, ticker, order_value, list_tier, port_value
    need = ["date", "ticker", "order_value", "list_tier", "port_value"]
    missing = [c for c in need if c not in od.columns]
    if missing:
        raise SystemExit(f"orders_daily.parquet missing columns: {missing}")

    od["date"] = pd.to_datetime(od["date"])
    od["band_pct"] = od["list_tier"].map(bands).fillna(bands.get("L3", 1.8))
    od["limit_value"] = (od["band_pct"] / 100.0) * od["port_value"]
    od["util_pct_before"] = (od["order_value"] / od["port_value"]) * 100.0
    od["was_violation"] = od["order_value"] > od["limit_value"]

    # Throttle: cap to band limit
    od["order_value_throttled"] = np.minimum(od["order_value"], od["limit_value"])
    od["util_pct_after"] = (od["order_value_throttled"] / od["port_value"]) * 100.0
    od["is_violation_after"] = od["order_value_throttled"] > od["limit_value"]

    # Save throttled orders (parquet for pipeline, csv for viewing)
    out_parquet = REPORTS / "orders_daily_throttled.parquet"
    out_csv = REPORTS / "orders_daily_throttled.csv"
    od[["date", "ticker", "order_value_throttled", "list_tier", "port_value"]].rename(
        columns={"order_value_throttled": "order_value"}
    ).to_parquet(out_parquet, index=False)
    od.to_csv(out_csv, index=False)

    # Canary log: summarize violations before/after
    before = int(od["was_violation"].sum())
    after = int(od["is_violation_after"].sum())
    viol_by_ticker = (
        od.loc[od["was_violation"], "ticker"].value_counts().head(10).to_dict()
    )
    canary = {
        "violations_before": before,
        "violations_after": after,
        "top10_violators_before": viol_by_ticker,
        "recommendation": "If 'after' > 0, raise bands slightly for affected tier(s) or reduce wiggle in bootstrap.",
    }
    canary_path = REPORTS / "canary_log.csv"
    # write a readable CSV view
    od[
        [
            "date",
            "ticker",
            "list_tier",
            "port_value",
            "order_value",
            "band_pct",
            "limit_value",
            "util_pct_before",
            "was_violation",
            "order_value_throttled",
            "util_pct_after",
            "is_violation_after",
        ]
    ].to_csv(canary_path, index=False)

    print(json.dumps(canary, indent=2))

    # OPEN the CSVs for review
    open_win(out_csv)
    open_win(canary_path)


if __name__ == "__main__":
    main()

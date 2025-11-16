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
        raise SystemExit(f"Missing {orders_p} — run your bootstrap first.")
    orders = pd.read_parquet(orders_p)
    # Expect [date, ticker, order_value, list_tier, port_value]
    for need in ["date", "ticker", "order_value", "list_tier", "port_value"]:
        if need not in orders.columns:
            raise SystemExit(f"orders_daily.parquet missing column: {need}")

    orders["date"] = pd.to_datetime(orders["date"])
    # Utilization of the daily turnover band: order_value / port_value * 100 (pct of book)
    orders["util_pct"] = (orders["order_value"] / orders["port_value"]) * 100.0
    # Summary by tier
    tiers = orders["list_tier"].fillna("NA").unique().tolist()
    rows = []
    for tier in sorted(tiers):
        df = orders[orders["list_tier"] == tier]
        if df.empty:
            continue
        rows.append(
            {
                "list_tier": tier,
                "n_rows": int(df.shape[0]),
                "p50_util_pct": float(df["util_pct"].quantile(0.50)),
                "p90_util_pct": float(df["util_pct"].quantile(0.90)),
                "p95_util_pct": float(df["util_pct"].quantile(0.95)),
                "p975_util_pct": float(df["util_pct"].quantile(0.975)),
                "p99_util_pct": float(df["util_pct"].quantile(0.99)),
                "max_util_pct": float(df["util_pct"].max()),
            }
        )
    summ = pd.DataFrame(rows).sort_values("list_tier")
    out_summary = REPORTS / "band_utilization_summary.csv"
    summ.to_csv(out_summary, index=False)

    # Detailed violations with current policy (if present)
    policy = load_yaml(CONFIG) if CONFIG.exists() else {}
    bands = policy.get("turnover_bands_pct_per_day", {"L1": 0.8, "L2": 1.2, "L3": 1.8})
    orders["band_pct_current"] = (
        orders["list_tier"].map(bands).fillna(bands.get("L3", 1.8))
    )
    orders["limit_value_current"] = (orders["band_pct_current"] / 100.0) * orders[
        "port_value"
    ]
    orders["violation_current"] = orders["order_value"] > orders["limit_value_current"]
    out_detail = REPORTS / "pretrade_violations_detailed.csv"
    orders.to_csv(out_detail, index=False)

    # Data-driven recommended bands (round up to 0.05% granularity, cap at 2.0%)
    rec = []
    for tier in sorted(tiers):
        df = orders[orders["list_tier"] == tier]
        if df.empty:
            continue
        p975 = float(df["util_pct"].quantile(0.975))  # 97.5th percentile
        # add 10% headroom, then cap to 2.0%
        target = min(p975 * 1.10, 2.0)
        # round up to 0.05 steps
        step = 0.05
        target = step * float(np.ceil(target / step))
        rec.append({"list_tier": tier, "recommended_band_pct": round(target, 2)})
    rec_df = pd.DataFrame(rec).sort_values("list_tier")
    out_rec = REPORTS / "band_recommendations.csv"
    rec_df.to_csv(out_rec, index=False)

    print(
        json.dumps(
            {
                "summary_csv": str(out_summary),
                "detail_csv": str(out_detail),
                "recommendations_csv": str(out_rec),
                "recommendations": rec_df.to_dict(orient="records"),
            },
            indent=2,
        )
    )

    # Open the three CSVs
    open_win(out_summary)
    open_win(out_detail)
    open_win(out_rec)
    # Also open policy for quick compare
    if CONFIG.exists():
        open_win(CONFIG)


if __name__ == "__main__":
    main()

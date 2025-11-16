from __future__ import annotations

import csv
import datetime as dt
import json
from pathlib import Path

import pandas as pd

ROOT = Path(r"F:\Projects\trading_stack_py")
DOCS = ROOT / "docs"
REPORTS = ROOT / "reports"
TRACKER = DOCS / "living_tracker.csv"
SUMMARY = REPORTS / "wk25_exec_engineering.csv"


def main():
    DOCS.mkdir(parents=True, exist_ok=True)
    if SUMMARY.exists():
        row = pd.read_csv(SUMMARY).iloc[0].to_dict()
    else:
        row = {}

    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fields = [
        "ts",
        "session",
        "artifact",
        "detail_csv",
        "style_summary_csv",
        "summary_csv",
        "orders_evaluated",
        "notional_inr",
        "best_total_tca_inr",
        "best_style_portfolio_hint",
        "git_sha8",
    ]

    TRACKER.parent.mkdir(parents=True, exist_ok=True)
    write_header = not TRACKER.exists()
    with TRACKER.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        w.writerow(
            {
                "ts": now,
                "session": "S-W25",
                "artifact": "W25 Exec Engineering",
                "detail_csv": str(row.get("detail_csv", "")),
                "style_summary_csv": str(row.get("style_summary_csv", "")),
                "summary_csv": str(SUMMARY),
                "orders_evaluated": row.get("orders_evaluated", ""),
                "notional_inr": row.get("notional_inr", ""),
                "best_total_tca_inr": row.get("best_total_tca_inr", ""),
                "best_style_portfolio_hint": row.get("best_style_portfolio_hint", ""),
                "git_sha8": "",  # optional: fill via your git helper if desired
            }
        )

    print(json.dumps({"tracker_csv": str(TRACKER), "session": "S-W25"}, indent=2))


if __name__ == "__main__":
    main()

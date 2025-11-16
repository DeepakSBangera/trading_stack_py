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
SUMMARY = REPORTS / "wk26_ops_cvar.csv"


def main():
    DOCS.mkdir(parents=True, exist_ok=True)
    row = pd.read_csv(SUMMARY).iloc[0].to_dict() if SUMMARY.exists() else {}
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    fields = [
        "ts",
        "session",
        "artifact",
        "as_of",
        "days_used",
        "VaR_95_pct",
        "CVaR_95_pct",
        "VaR_99_pct",
        "CVaR_99_pct",
        "ret_series_csv",
        "stress_table_csv",
        "git_sha8",
    ]
    write_header = not TRACKER.exists()
    with TRACKER.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        w.writerow(
            {
                "ts": now,
                "session": "S-W26",
                "artifact": "W26 Ops CVaR & Stress",
                "as_of": row.get("as_of", ""),
                "days_used": row.get("days_used", ""),
                "VaR_95_pct": row.get("VaR_95_pct", ""),
                "CVaR_95_pct": row.get("CVaR_95_pct", ""),
                "VaR_99_pct": row.get("VaR_99_pct", ""),
                "CVaR_99_pct": row.get("CVaR_99_pct", ""),
                "ret_series_csv": row.get("ret_series_csv", ""),
                "stress_table_csv": row.get("stress_table_csv", ""),
                "git_sha8": "",
            }
        )
    print(json.dumps({"tracker_csv": str(TRACKER), "session": "S-W26"}, indent=2))


if __name__ == "__main__":
    main()

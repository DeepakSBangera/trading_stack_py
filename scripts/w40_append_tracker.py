from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

ROOT = Path(r"F:\Projects\trading_stack_py")
DOCS = ROOT / "docs"
TRACKER = DOCS / "living_tracker.csv"


def ensure_header():
    DOCS.mkdir(parents=True, exist_ok=True)
    if not TRACKER.exists():
        with open(TRACKER, "w", encoding="utf-8", newline="") as f:
            f.write("date,session,hours,artifacts,gates,risks,decisions\n")


def main():
    ensure_header()
    row = [
        datetime.now().strftime("%Y-%m-%d %H:%M"),
        "S-W40",
        "4",
        "wk40_exec_quality.csv; wk40_exec_quality_summary.json",
        "exec schedule benchmarking added",
        "none",
        "adopt per-ticker best style; review small-ADV names",
    ]
    with open(TRACKER, "a", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(row)
    print({"tracker_csv": str(TRACKER), "session": "S-W40"})


if __name__ == "__main__":
    main()

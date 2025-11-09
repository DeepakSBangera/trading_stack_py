from __future__ import annotations
from pathlib import Path
from datetime import datetime
import csv

ROOT = Path(r"F:\Projects\trading_stack_py")
DOCS = ROOT / "docs"
TRACKER = DOCS / "living_tracker.csv"

def ensure_header():
    DOCS.mkdir(parents=True, exist_ok=True)
    if not TRACKER.exists():
        with open(TRACKER, "w", newline="", encoding="utf-8") as f:
            f.write("date,session,hours,artifacts,gates,risks,decisions\n")

def main():
    ensure_header()
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    row = [
        now,
        "S-W39",
        "4",  # hours
        "wk39_capacity_audit.csv; wk39_capacity_summary.json; wk39_capacity_stress.csv",
        "capacity caps drafted",
        "none",
        "adopt per-order %ADV cap; sector/name caps noted",
    ]
    with open(TRACKER, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)
    print({"tracker_csv": str(TRACKER), "session": "S-W39"})

if __name__ == "__main__":
    main()


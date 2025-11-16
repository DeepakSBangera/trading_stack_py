from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

ROOT = Path(r"F:\Projects\trading_stack_py")
DOCS = ROOT / "docs"
TRACKER = DOCS / "living_tracker.csv"


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    TRACKER.touch(exist_ok=True)
    row = {
        "date_ist": pd.Timestamp.utcnow()
        .tz_convert("Asia/Kolkata")
        .strftime("%Y-%m-%d %H:%M"),
        "session": "S-W44",
        "notes": "Red-Team & DR drill: existence/manifest/backup checks; snapshot+MD saved.",
    }
    hdr = ["date_ist", "session", "notes"]
    write_header = TRACKER.stat().st_size == 0
    with open(TRACKER, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=hdr)
        if write_header:
            w.writeheader()
        w.writerow(row)
    print({"tracker_csv": str(TRACKER), "session": row["session"]})


if __name__ == "__main__":
    main()

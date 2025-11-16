from __future__ import annotations

import csv
import time
from pathlib import Path

ROOT = Path(r"F:\Projects\trading_stack_py")
TRACKER = ROOT / "docs" / "living_tracker.csv"


def main() -> None:
    TRACKER.parent.mkdir(parents=True, exist_ok=True)
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    row = [now, "S-W41", "Momentum tilt (12/6/1m), inverse-vol weights, top-30."]
    write_header = not TRACKER.exists()
    with open(TRACKER, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["ts", "session", "note"])
        w.writerow(row)
    print({"tracker_csv": str(TRACKER), "session": "S-W41"})


if __name__ == "__main__":
    main()

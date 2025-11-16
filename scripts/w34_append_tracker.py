from __future__ import annotations

import csv
import datetime as dt
import json
from pathlib import Path

ROOT = Path(r"F:\Projects\trading_stack_py")
DOCS = ROOT / "docs"
REPORTS = ROOT / "reports"
TRACKER = DOCS / "living_tracker.csv"
SUMMARY = REPORTS / "w34_leverage_summary.json"
RESULTS = REPORTS / "wk34_leverage_throttle.csv"


def main():
    DOCS.mkdir(parents=True, exist_ok=True)
    rows = 0
    on_days = 0
    if SUMMARY.exists():
        j = json.loads(SUMMARY.read_text(encoding="utf-8"))
        rows = j.get("rows", 0)
        on_days = j.get("on_days", 0)

    fields = [
        "ts",
        "session",
        "artifact",
        "as_of",
        "results_csv",
        "best_note",
        "git_sha8",
    ]
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    write_header = not TRACKER.exists()
    with TRACKER.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        w.writerow(
            {
                "ts": now,
                "session": "S-W34",
                "artifact": "W34 Safe Leverage Switch",
                "as_of": "",
                "results_csv": str(RESULTS) if RESULTS.exists() else "",
                "best_note": f"rows={rows}, on_days={on_days}",
                "git_sha8": "",
            }
        )
    print(
        {
            "tracker_csv": str(TRACKER),
            "session": "S-W34",
            "rows": rows,
            "on_days": on_days,
        }
    )


if __name__ == "__main__":
    main()

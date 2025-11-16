from __future__ import annotations

import csv
import datetime as dt
import json
from pathlib import Path

ROOT = Path(r"F:\Projects\trading_stack_py")
DOCS = ROOT / "docs"
REPORTS = ROOT / "reports"
TRACKER = DOCS / "living_tracker.csv"
SUMMARY = REPORTS / "w33_barbell_summary.json"
RESULTS = REPORTS / "w33_barbell_results.csv"


def main():
    DOCS.mkdir(parents=True, exist_ok=True)
    rows = 0
    note = ""
    if SUMMARY.exists():
        j = json.loads(SUMMARY.read_text(encoding="utf-8"))
        rows = j.get("rows", 0)
        splits = j.get("splits_effective", {})
        cap = j.get("per_name_cap", "")
        note = f"Barbell rows={rows}, split={splits}, cap={cap}"

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
                "session": "S-W33",
                "artifact": "W33 Barbell Combine",
                "as_of": "",
                "results_csv": str(RESULTS) if RESULTS.exists() else "",
                "best_note": note,
                "git_sha8": "",
            }
        )
    print(
        json.dumps(
            {"tracker_csv": str(TRACKER), "session": "S-W33", "rows": rows}, indent=2
        )
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import csv
import datetime as dt
import json
from pathlib import Path

ROOT = Path(r"F:\Projects\trading_stack_py")
DOCS = ROOT / "docs"
REPORTS = ROOT / "reports"
TRACKER = DOCS / "living_tracker.csv"
SUMMARY = REPORTS / "w37_alpha_table_summary.json"


def main():
    DOCS.mkdir(parents=True, exist_ok=True)
    promote, retire = [], []
    if SUMMARY.exists():
        j = json.loads(SUMMARY.read_text(encoding="utf-8"))
        promote = j.get("promote", []) or []
        retire = j.get("retire", []) or []

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
    best_note = f"Promote: {', '.join(promote) if promote else '—'} | Retire: {', '.join(retire) if retire else '—'}"
    write_header = not TRACKER.exists()
    with TRACKER.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        w.writerow(
            {
                "ts": now,
                "session": "S-W37",
                "artifact": "W37 Alpha Table & Promotion Ladder",
                "as_of": "",
                "results_csv": str(REPORTS / "wk37_alpha_table.csv"),
                "best_note": best_note,
                "git_sha8": "",
            }
        )
    print({"tracker_csv": str(TRACKER), "session": "S-W37", "note": best_note})


if __name__ == "__main__":
    main()

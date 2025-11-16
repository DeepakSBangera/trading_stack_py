from __future__ import annotations

import csv
import datetime as dt
import json
from pathlib import Path

ROOT = Path(r"F:\Projects\trading_stack_py")
DOCS = ROOT / "docs"
REPORTS = ROOT / "reports"
TRACKER = DOCS / "living_tracker.csv"
SUMMARY = REPORTS / "w35_tca_summary.json"
GRID = REPORTS / "wk35_tca_tuning.csv"


def main():
    DOCS.mkdir(parents=True, exist_ok=True)
    best_note = ""
    if SUMMARY.exists():
        j = json.loads(SUMMARY.read_text(encoding="utf-8"))
        b = j.get("best")
        if b:
            best_note = (
                f"best: entry={b.get('entry_pp'):.4%}, exit={b.get('exit_pp'):.4%}, "
                f"hold={int(b.get('min_hold_days', 0))}d, "
                f"turnover={b.get('turnover', 0):.4f}, "
                f"tca_inr={b.get('tca_inr', 0):,.0f}"
            )
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
                "session": "S-W35",
                "artifact": "W35 Turnover & Hysteresis Tuning",
                "as_of": "",
                "results_csv": str(GRID) if GRID.exists() else "",
                "best_note": best_note,
                "git_sha8": "",
            }
        )
    print({"tracker_csv": str(TRACKER), "session": "S-W35", "best_note": best_note})


if __name__ == "__main__":
    main()

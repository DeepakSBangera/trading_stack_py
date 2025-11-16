from __future__ import annotations

import csv
import datetime as dt
import json
from pathlib import Path

ROOT = Path(r"F:\Projects\trading_stack_py")
DOCS = ROOT / "docs"
REPORTS = ROOT / "reports"
TRACKER = DOCS / "living_tracker.csv"
DIAG = REPORTS / "w32_offline_rl_diag.json"
OUTCSV = REPORTS / "w32_offline_rl_sizing.csv"


def main():
    DOCS.mkdir(parents=True, exist_ok=True)
    note = ""
    if DIAG.exists():
        j = json.loads(DIAG.read_text(encoding="utf-8"))
        note = f"W32 offline RL sizing: train_rows={j.get('train_rows')}, current_rows={j.get('current_rows')}, cap={j.get('per_name_cap')}"
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
                "session": "S-W32",
                "artifact": "W32 Offline RL Sizing",
                "as_of": "",
                "results_csv": str(OUTCSV) if OUTCSV.exists() else "",
                "best_note": note,
                "git_sha8": "",
            }
        )
    print(json.dumps({"tracker_csv": str(TRACKER), "session": "S-W32"}, indent=2))


if __name__ == "__main__":
    main()

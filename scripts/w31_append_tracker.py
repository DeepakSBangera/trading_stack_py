from __future__ import annotations

import csv
import datetime as dt
import json
from pathlib import Path

ROOT = Path(r"F:\Projects\trading_stack_py")
DOCS = ROOT / "docs"
REPORTS = ROOT / "reports"
TRACKER = DOCS / "living_tracker.csv"
DIAG = REPORTS / "w31_exec_bandit_diag.json"
ASSIGN = REPORTS / "w31_bandit_assignments.csv"


def main():
    DOCS.mkdir(parents=True, exist_ok=True)
    note = ""
    if DIAG.exists():
        j = json.loads(DIAG.read_text(encoding="utf-8"))
        orders = j.get("orders_in", 0)
        hist = j.get("history_rows", 0)
        note = f"Exec bandit assigned {orders} orders; history_rows={hist}; arms=VWAP/TWAP/POV10/POV20; ε={j.get('epsilon')}"

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
                "session": "S-W31",
                "artifact": "W31 Execution Bandit",
                "as_of": "",
                "results_csv": str(ASSIGN) if ASSIGN.exists() else "",
                "best_note": note,
                "git_sha8": "",
            }
        )
    print(json.dumps({"tracker_csv": str(TRACKER), "session": "S-W31"}, indent=2))


if __name__ == "__main__":
    main()

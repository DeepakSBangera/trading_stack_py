from __future__ import annotations

import csv
import datetime as dt
import json
from pathlib import Path

import pandas as pd

ROOT = Path(r"F:\Projects\trading_stack_py")
DOCS = ROOT / "docs"
REPORTS = ROOT / "reports"
TRACKER = DOCS / "living_tracker.csv"
POLICY = REPORTS / "w27_bandit_selection.csv"
SUMMARY = REPORTS / "w27_bandit_summary.json"


def main():
    DOCS.mkdir(parents=True, exist_ok=True)
    row = {}
    if SUMMARY.exists():
        try:
            row = json.loads(SUMMARY.read_text(encoding="utf-8"))
        except Exception:
            row = {}
    pol = pd.read_csv(POLICY) if POLICY.exists() else pd.DataFrame()
    weights_note = (
        "; ".join([f"{r.sleeve}={r.weight:.3f}" for r in pol.itertuples()])
        if not pol.empty
        else ""
    )

    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fields = [
        "ts",
        "session",
        "artifact",
        "as_of",
        "sleeves",
        "policy_csv",
        "weights",
        "git_sha8",
    ]
    write_header = not TRACKER.exists()
    with TRACKER.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        w.writerow(
            {
                "ts": now,
                "session": "S-W27",
                "artifact": "W27 Bandit Selection",
                "as_of": row.get("as_of", ""),
                "sleeves": (
                    ", ".join(row.get("sleeves", []))
                    if isinstance(row.get("sleeves", []), list)
                    else ""
                ),
                "policy_csv": row.get("policy_csv", ""),
                "weights": weights_note,
                "git_sha8": "",
            }
        )
    print(json.dumps({"tracker_csv": str(TRACKER), "session": "S-W27"}, indent=2))


if __name__ == "__main__":
    main()

# scripts/w14_append_tracker.py
from __future__ import annotations

import csv
import datetime as dt
import json
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(r"F:\Projects\trading_stack_py")
REPORTS = ROOT / "reports"
DOCS = ROOT / "docs"
TRACKER = DOCS / "living_tracker.csv"

POS = REPORTS / "w14_positions_reco.csv"
DRIFT = REPORTS / "w14_drift_summary.csv"

SESSION = "S-W14"
NOW_ISO = dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _git_sha8():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "nogit"


def _stats():
    out = {}
    if DRIFT.exists():
        d = pd.read_csv(DRIFT)
        for k in ["names", "gross_notional_inr", "abs_drift_sum", "abs_drift_p90"]:
            if k in d.columns:
                out[k] = float(d.iloc[0][k])
    return out


def main():
    DOCS.mkdir(parents=True, exist_ok=True)
    sha = _git_sha8()
    s = _stats()
    artifacts = ";".join(map(str, [POS, DRIFT]))
    header = [
        "timestamp",
        "session",
        "branch",
        "git_sha8",
        "artifacts",
        "names",
        "gross_notional_inr",
        "abs_drift_sum",
        "abs_drift_p90",
        "notes",
    ]
    row = [
        NOW_ISO,
        SESSION,
        "main",
        sha,
        artifacts,
        s.get("names"),
        s.get("gross_notional_inr"),
        s.get("abs_drift_sum"),
        s.get("abs_drift_p90"),
        "W14: T+1 reconciliation & drift",
    ]
    write_header = not TRACKER.exists()
    with TRACKER.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow(row)
    print(
        json.dumps(
            {
                "tracker_csv": str(TRACKER),
                "session": SESSION,
                "git_sha8": sha,
                "stats": s,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

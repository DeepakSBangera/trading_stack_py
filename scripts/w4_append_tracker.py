# scripts/w4_append_tracker.py
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

VOLSTOPS = REPORTS / "wk4_voltarget_stops.csv"
THROTTLE = REPORTS / "dd_throttle_map.csv"
KILLSW = REPORTS / "kill_switch.yaml"

SESSION = "S-W4"
NOW_ISO = dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _git_sha8() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
        return out
    except Exception:
        return "nogit"


def _stats():
    s = {}
    if VOLSTOPS.exists():
        v = pd.read_csv(VOLSTOPS)
        atr_cols = [c for c in v.columns if c.startswith("atr_")]
        s["tickers"] = v["ticker"].nunique()
        s["rows"] = int(v.shape[0])
        if atr_cols:
            c = atr_cols[0]
            s["with_atr"] = int((~v[c].isna()).sum())
    return s


def main():
    DOCS.mkdir(parents=True, exist_ok=True)
    sha8 = _git_sha8()
    s = _stats()
    artifacts = ";".join([str(VOLSTOPS), str(THROTTLE), str(KILLSW)])

    header = [
        "timestamp",
        "session",
        "branch",
        "git_sha8",
        "artifacts",
        "tickers",
        "rows",
        "with_atr",
        "notes",
    ]
    row = [
        NOW_ISO,
        SESSION,
        "main",
        sha8,
        artifacts,
        s.get("tickers"),
        s.get("rows"),
        s.get("with_atr"),
        "W4: vol target + ATR stops + kill-switch + DD throttle",
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
                "git_sha8": sha8,
                "stats": s,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

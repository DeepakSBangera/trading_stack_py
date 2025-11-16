# scripts/w6_append_tracker.py
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

COMPARE = REPORTS / "wk6_portfolio_compare.csv"
EXPO = REPORTS / "factor_exposure_weekly.csv"
CAP = REPORTS / "capacity_curve.csv"
WE_CAP = REPORTS / "wk6_weights_capped.csv"
VAL = REPORTS / "wk6_caps_validation.csv"

SESSION = "S-W6"
NOW_ISO = dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _git_sha8():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "nogit"


def _stats():
    s = {}
    if COMPARE.exists():
        c = pd.read_csv(COMPARE)
        s["schemes"] = c["scheme"].nunique() if "scheme" in c.columns else None
        s["herf_median"] = (
            float(c["herfindahl"].median()) if "herfindahl" in c.columns else None
        )
    if CAP.exists():
        cp = pd.read_csv(CAP)
        s["capacity_points"] = int(cp.shape[0])
    return s


def main():
    DOCS.mkdir(parents=True, exist_ok=True)
    sha = _git_sha8()
    artifacts = ";".join([str(COMPARE), str(EXPO), str(CAP), str(WE_CAP), str(VAL)])
    s = _stats()
    header = [
        "timestamp",
        "session",
        "branch",
        "git_sha8",
        "artifacts",
        "schemes",
        "herf_median",
        "capacity_points",
        "notes",
    ]
    row = [
        NOW_ISO,
        SESSION,
        "main",
        sha,
        artifacts,
        s.get("schemes"),
        s.get("herf_median"),
        s.get("capacity_points"),
        "W6: optimizer compare + exposures + capacity + caps",
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

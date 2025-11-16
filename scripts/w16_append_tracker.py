from __future__ import annotations

import csv
import datetime as dt
import json
import subprocess
from pathlib import Path

ROOT = Path(r"F:\Projects\trading_stack_py")
REPORTS = ROOT / "reports"
DOCS = ROOT / "docs"
TRACKER = DOCS / "living_tracker.csv"

PROP = REPORTS / "w16_caps_property_tests.csv"
EDGE = REPORTS / "w16_caps_edge_cases.csv"
DIAG = REPORTS / "w16_caps_diag.json"

SESSION = "S-W16"
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
    if DIAG.exists():
        d = json.loads(DIAG.read_text(encoding="utf-8"))
        for k in ["per_name_breaches", "sector_breaches", "per_name_cap", "sector_cap"]:
            s[k] = d.get(k)
    return s


def main():
    DOCS.mkdir(parents=True, exist_ok=True)
    sha = _git_sha8()
    s = _stats()
    artifacts = ";".join(map(str, [PROP, EDGE, DIAG]))
    header = [
        "timestamp",
        "session",
        "branch",
        "git_sha8",
        "artifacts",
        "per_name_cap",
        "sector_cap",
        "per_name_breaches",
        "sector_breaches",
        "notes",
    ]
    row = [
        NOW_ISO,
        SESSION,
        "main",
        sha,
        artifacts,
        s.get("per_name_cap"),
        s.get("sector_cap"),
        s.get("per_name_breaches"),
        s.get("sector_breaches"),
        "W16: caps property & edge tests",
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

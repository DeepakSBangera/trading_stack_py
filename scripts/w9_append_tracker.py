# scripts/w9_append_tracker.py
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

SCHEMA = REPORTS / "pit_schema_audit.csv"
MONO = REPORTS / "pit_monotonic_audit.csv"
SURV = REPORTS / "universe_survivorship_audit.csv"

SESSION = "S-W9"
NOW_ISO = dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _git_sha8():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "nogit"


def _stats():
    out = {}
    if SCHEMA.exists():
        s = pd.read_csv(SCHEMA)
        out["tickers_checked"] = int(s.shape[0])
        out["pit_suspect_cols_any"] = (
            int(s["pit_suspect_cols"].astype(str).str.len().gt(0).sum()) if "pit_suspect_cols" in s else 0
        )
        out["missing_date_or_close"] = int((s["issue"] == "no_date_or_close").sum()) if "issue" in s else 0
    if MONO.exists():
        m = pd.read_csv(MONO)
        out["mono_fail"] = int((~m["is_monotonic_increasing"]).sum()) if "is_monotonic_increasing" in m else 0
        out["dup_dates_sum"] = int(m["duplicate_dates"].sum()) if "duplicate_dates" in m else 0
        out["gap_windows_sum"] = int(m["gap_windows_gt5d"].sum()) if "gap_windows_gt5d" in m else 0
    if SURV.exists():
        u = pd.read_csv(SURV)
        out["dates_checked"] = int(u.shape[0])
        out["missing_any_days"] = int((u["missing"] > 0).sum()) if "missing" in u else 0
    return out


def main():
    DOCS.mkdir(parents=True, exist_ok=True)
    sha = _git_sha8()
    s = _stats()
    artifacts = ";".join(map(str, [SCHEMA, MONO, SURV]))
    header = [
        "timestamp",
        "session",
        "branch",
        "git_sha8",
        "artifacts",
        "tickers_checked",
        "pit_suspect_cols_any",
        "missing_date_or_close",
        "mono_fail",
        "dup_dates_sum",
        "gap_windows_sum",
        "dates_checked",
        "missing_any_days",
        "notes",
    ]
    row = [
        NOW_ISO,
        SESSION,
        "main",
        sha,
        artifacts,
        s.get("tickers_checked"),
        s.get("pit_suspect_cols_any"),
        s.get("missing_date_or_close"),
        s.get("mono_fail"),
        s.get("dup_dates_sum"),
        s.get("gap_windows_sum"),
        s.get("dates_checked"),
        s.get("missing_any_days"),
        "W9: PIT schema + monotonic + survivorship",
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

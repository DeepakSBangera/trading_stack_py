# scripts/w13_append_tracker.py
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

TCA_CSV = REPORTS / "wk13_tca_summary.csv"
FILLS_CSV = REPORTS / "wk13_dryrun_fills.csv"

SESSION = "S-W13"
NOW_ISO = dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _git_sha8() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
        return out
    except Exception:
        return "nogit"


def _read_tca():
    if not TCA_CSV.exists():
        return {}
    import pandas as pd

    df = pd.read_csv(TCA_CSV)
    # daily roll-up already present; also compute overall
    orders = int(df["orders"].sum()) if "orders" in df.columns else None
    tca_cost = float(df["tca_cost"].sum()) if "tca_cost" in df.columns else None
    med_slip = float(df["med_slip_bps"].median()) if "med_slip_bps" in df.columns else None
    p90_slip = float(df["p90_slip_bps"].median()) if "p90_slip_bps" in df.columns else None
    return {
        "orders": orders,
        "tca_cost_inr": tca_cost,
        "med_slip_bps": med_slip,
        "p90_slip_bps": p90_slip,
    }


def main():
    DOCS.mkdir(parents=True, exist_ok=True)
    stats = _read_tca()
    sha8 = _git_sha8()
    artifacts = ";".join([str(FILLS_CSV), str(TCA_CSV)])

    header = [
        "timestamp",
        "session",
        "branch",
        "git_sha8",
        "artifacts",
        "orders",
        "tca_cost_inr",
        "med_slip_bps",
        "p90_slip_bps",
        "notes",
    ]
    row = [
        NOW_ISO,
        SESSION,
        "main",
        sha8,
        artifacts,
        stats.get("orders"),
        stats.get("tca_cost_inr"),
        stats.get("med_slip_bps"),
        stats.get("p90_slip_bps"),
        "W13: dry-run fills + TCA",
    ]

    write_header = not TRACKER.exists()
    with TRACKER.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow(row)

    out = {
        "tracker_csv": str(TRACKER),
        "session": SESSION,
        "git_sha8": sha8,
        "artifacts": [str(FILLS_CSV), str(TCA_CSV)],
        "stats": stats,
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

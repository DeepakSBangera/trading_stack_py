# scripts/w15_append_tracker.py
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

CURVES = REPORTS / "w15_exec_curves_by_ticker.csv"
SUMM = REPORTS / "w15_exec_summary.csv"

SESSION = "S-W15"
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
    if CURVES.exists():
        c = pd.read_csv(CURVES)
        s["curve_rows"] = int(c.shape[0])
        s["tickers"] = int(c["ticker"].nunique()) if "ticker" in c.columns else None
    if SUMM.exists():
        d = pd.read_csv(SUMM)
        port = d[d["ticker"] == "__PORTFOLIO__"]
        if not port.empty:
            s["portfolio_best_pov"] = float(port.iloc[0]["best_pov_pct"])
            s["portfolio_exp_tca_bps"] = float(port.iloc[0]["best_exp_tca_bps"])
    return s


def main():
    DOCS.mkdir(parents=True, exist_ok=True)
    sha = _git_sha8()
    s = _stats()
    artifacts = ";".join(map(str, [CURVES, SUMM]))
    header = [
        "timestamp",
        "session",
        "branch",
        "git_sha8",
        "artifacts",
        "curve_rows",
        "tickers",
        "portfolio_best_pov",
        "portfolio_exp_tca_bps",
        "notes",
    ]
    row = [
        NOW_ISO,
        SESSION,
        "main",
        sha,
        artifacts,
        s.get("curve_rows"),
        s.get("tickers"),
        s.get("portfolio_best_pov"),
        s.get("portfolio_exp_tca_bps"),
        "W15: Broker sim ADV/POV slippage curves",
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

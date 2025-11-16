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
RESULTS = REPORTS / "w28_ope_results.csv"
SUMMARY = REPORTS / "w28_ope_summary.json"


def main():
    DOCS.mkdir(parents=True, exist_ok=True)

    # read best DR policy line (if exists)
    pol_summary = ""
    top_line = {}
    if RESULTS.exists():
        df = pd.read_csv(RESULTS)
        # Prefer DR, then SNIPS
        pref = df[df["method"] == "DR"].copy()
        if pref.empty:
            pref = df.copy()
        pref = pref.sort_values("estimate_bps_per_day", ascending=False)
        if not pref.empty:
            top_line = pref.iloc[0].to_dict()
            pol_summary = f"{top_line.get('policy', '')}/{top_line.get('method', '')}: {top_line.get('estimate_bps_per_day', 0)} bps/d (~{top_line.get('approx_annual_pct', 0)}%/yr)"

    row = {}
    if SUMMARY.exists():
        try:
            row = json.loads(SUMMARY.read_text(encoding="utf-8"))
        except Exception:
            row = {}

    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fields = [
        "ts",
        "session",
        "artifact",
        "as_of",
        "results_csv",
        "best_note",
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
                "session": "S-W28",
                "artifact": "W28 Off-Policy Evaluation",
                "as_of": row.get("as_of", ""),
                "results_csv": row.get("results_csv", ""),
                "best_note": pol_summary,
                "git_sha8": "",
            }
        )
    print(json.dumps({"tracker_csv": str(TRACKER), "session": "S-W28"}, indent=2))


if __name__ == "__main__":
    main()

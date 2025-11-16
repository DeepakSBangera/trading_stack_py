from __future__ import annotations

import csv
import datetime as dt
import json
from pathlib import Path

ROOT = Path(r"F:\Projects\trading_stack_py")
DOCS = ROOT / "docs"
REPORTS = ROOT / "reports"
TRACKER = DOCS / "living_tracker.csv"
SUMMARY = REPORTS / "w36_overlay_summary.json"
GRID = REPORTS / "wk36_options_overlay.csv"


def main():
    DOCS.mkdir(parents=True, exist_ok=True)
    best_note = ""
    if SUMMARY.exists():
        j = json.loads(SUMMARY.read_text(encoding="utf-8"))
        b = j.get("best")
        if b:
            if b.get("kind") == "PUT":
                best_note = (
                    f"PUT hedge={b.get('hedge_ratio', 0):.0%}, K={b.get('moneyness', 1.0):.0%}, "
                    f"tenor={int(b.get('tenor_days', 30))}d, "
                    f"cost={b.get('cost_bps_ann', 0):.0f}bps/yr, "
                    f"DD {b.get('base_max_dd', 0):.2%}→{b.get('overlay_max_dd', 0):.2%}"
                )
            else:
                best_note = (
                    f"CALL cov={b.get('call_coverage', 0):.0%}, K={b.get('moneyness', 1.0):.0%}, "
                    f"tenor={int(b.get('tenor_days', 30))}d, "
                    f"DD {b.get('base_max_dd', 0):.2%}→{b.get('overlay_max_dd', 0):.2%}"
                )
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
                "session": "S-W36",
                "artifact": "W36 Options Overlay",
                "as_of": "",
                "results_csv": str(GRID) if GRID.exists() else "",
                "best_note": best_note,
                "git_sha8": "",
            }
        )
    print({"tracker_csv": str(TRACKER), "session": "S-W36", "best_note": best_note})


if __name__ == "__main__":
    main()

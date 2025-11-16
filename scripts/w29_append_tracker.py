from __future__ import annotations

import csv
import datetime as dt
import json
from pathlib import Path

ROOT = Path(r"F:\Projects\trading_stack_py")
DOCS = ROOT / "docs"
REPORTS = ROOT / "reports"
TRACKER = DOCS / "living_tracker.csv"
POLICY = REPORTS / "w29_safe_policy.csv"
DIAG = REPORTS / "w29_safety_diag.json"


def main():
    DOCS.mkdir(parents=True, exist_ok=True)

    note = ""
    if DIAG.exists():
        j = json.loads(DIAG.read_text(encoding="utf-8"))
        src = "ACCEPTED" if j.get("accepted") else "BASELINE"
        gain = j.get("gain_dr_per_day_bps", 0.0)
        lo, hi = j.get("gain_dr_ci_bps") or [0.0, 0.0]
        note = f"SPIBB {src}; ΔDR ≈ {gain} bps/d (CI [{lo},{hi}])"

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
                "session": "S-W29",
                "artifact": "W29 Safe Policy (SPIBB-like)",
                "as_of": "",
                "results_csv": str(POLICY) if POLICY.exists() else "",
                "best_note": note,
                "git_sha8": "",
            }
        )

    print(json.dumps({"tracker_csv": str(TRACKER), "session": "S-W29"}, indent=2))


if __name__ == "__main__":
    main()

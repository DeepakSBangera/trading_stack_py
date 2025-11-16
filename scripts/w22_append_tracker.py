from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

ROOT = Path(r"F:\Projects\trading_stack_py")
DOCS = ROOT / "docs"
REPORTS = ROOT / "reports"
TRACK = DOCS / "living_tracker.csv"
SUMJ = REPORTS / "w22_summary.json"


def _git_sha_short() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT),
            stderr=subprocess.DEVNULL,
        )
        return out.decode("utf-8", "ignore").strip()
    except Exception:
        return "????????"


def main():
    if not SUMJ.exists():
        raise FileNotFoundError(f"{SUMJ} not found. Run w22_pit_integrity.py first.")

    with open(SUMJ, encoding="utf-8") as f:
        s = json.load(f)

    DOCS.mkdir(parents=True, exist_ok=True)
    write_header = not TRACK.exists()

    with open(TRACK, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["session", "module", "artifact", "rows", "sr_is", "sr_oos", "git_sha8"])
        w.writerow(
            [
                "S-W22",
                "w22_pit_integrity",
                s.get("out_summary_csv", ""),
                s.get("tickers", 0),
                "",  # sr_is (NA for W22)
                "",  # sr_oos (NA for W22)
                s.get("git_sha8") or _git_sha_short(),
            ]
        )

    print(
        {
            "tracker_csv": str(TRACK),
            "session": "S-W22",
            "tickers": s.get("tickers", 0),
            "pass_pit": s.get("pass_pit", 0),
            "fail_pit": s.get("fail_pit", 0),
        }
    )


if __name__ == "__main__":
    main()

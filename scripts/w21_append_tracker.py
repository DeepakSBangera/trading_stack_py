from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

ROOT = Path(r"F:\Projects\trading_stack_py")
DOCS = ROOT / "docs"
REPORTS = ROOT / "reports"
TRACK = DOCS / "living_tracker.csv"
W21_JSON = REPORTS / "w21_summary.json"


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
    if not W21_JSON.exists():
        raise FileNotFoundError(f"{W21_JSON} not found. Run w21_purgedcv_pbo.py first.")

    with open(W21_JSON, encoding="utf-8") as f:
        s = json.load(f)

    DOCS.mkdir(parents=True, exist_ok=True)
    write_header = not TRACK.exists()

    with open(TRACK, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["session", "module", "artifact", "rows", "sr_is", "sr_oos", "git_sha8"])
        w.writerow(
            [
                "S-W21",
                "w21_purgedcv_pbo",
                s.get("out_csv", ""),
                s.get("rows", 0),
                s.get("sr_is", 0.0),
                s.get("sr_oos", 0.0),
                s.get("git_sha8") or _git_sha_short(),
            ]
        )

    print(
        {
            "tracker_csv": str(TRACK),
            "session": "S-W21",
            "rows": s.get("rows", 0),
            "sr_is": s.get("sr_is", 0.0),
            "sr_oos": s.get("sr_oos", 0.0),
        }
    )


if __name__ == "__main__":
    main()

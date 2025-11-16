from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

ROOT = Path(r"F:\Projects\trading_stack_py")
DOCS = ROOT / "docs"
REPORTS = ROOT / "reports"
TRACK = DOCS / "living_tracker.csv"
SUMJ = REPORTS / "w23_factor_summary.json"


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
        raise FileNotFoundError(
            f"{SUMJ} not found. Run w23_factor_risk_model.py first."
        )
    with open(SUMJ, encoding="utf-8") as f:
        s = json.load(f)

    DOCS.mkdir(parents=True, exist_ok=True)
    write_header = not TRACK.exists()
    with open(TRACK, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(
                ["session", "module", "artifact", "rows", "sr_is", "sr_oos", "git_sha8"]
            )
        w.writerow(
            [
                "S-W23",
                "w23_factor_risk_model",
                s.get("detail_csv", ""),
                s.get("universe", 0),
                "",
                "",  # sr_is/sr_oos NA for W23
                s.get("git_sha8") or _git_sha_short(),
            ]
        )

    print(
        {
            "tracker_csv": str(TRACK),
            "session": "S-W23",
            "universe": s.get("universe", 0),
            "portfolio_z_exposures": s.get("portfolio_exposures_z", {}),
        }
    )


if __name__ == "__main__":
    main()

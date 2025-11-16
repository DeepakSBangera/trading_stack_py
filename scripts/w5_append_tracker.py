from __future__ import annotations

import datetime as dt
import os
import sys
from pathlib import Path

ROOT = Path(r"F:\Projects\trading_stack_py")
DOCS = ROOT / "docs"
TRACKER = DOCS / "living_tracker.csv"
REPORTS = ROOT / "reports"


def open_win(p: Path):
    if sys.platform.startswith("win") and p.exists():
        os.startfile(p)


def ensure_header():
    DOCS.mkdir(parents=True, exist_ok=True)
    if not TRACKER.exists():
        TRACKER.write_text(
            "date,session,hours,artifacts,gates,risks,decisions\n", encoding="utf-8"
        )


def main():
    ensure_header()
    today = dt.date.today().isoformat()
    artifacts = "; ".join(
        ["wk5_walkforward_dsr.csv", "w5_promote_retire.csv", "canary_log.csv"]
    )
    line = (
        f"{today},S-W5,~1.0h,"
        f'"{artifacts}",'
        f'"Gate: dsr_out>0.5 & sr_out>0; promote if true",'
        f'"Multiple-trials penalty via DSR; short sample caveat",'
        f'"Promote only if W6 risk caps also pass"\n'
    )
    with TRACKER.open("a", encoding="utf-8") as f:
        f.write(line)
    open_win(TRACKER)
    open_win(REPORTS)


if __name__ == "__main__":
    main()

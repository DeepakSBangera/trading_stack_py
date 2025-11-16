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
        TRACKER.write_text("date,session,hours,artifacts,gates,risks,decisions\n", encoding="utf-8")


def main():
    ensure_header()
    today = dt.date.today().isoformat()
    artifacts = "; ".join(["events_calendar.csv", "events_position_flags.csv", "event_rules.yaml"])
    gates = "Pre/post earnings freeze + reduced risk; holidays block"
    risks = "Synthetic calendar; refine with real vendor feed later"
    decisions = "Multiply risk by event risk_mult; disallow adds/rebalance as flagged"
    line = f'{today},S-W8,~1.0h,"{artifacts}","{gates}","{risks}","{decisions}"\n'
    with TRACKER.open("a", encoding="utf-8") as f:
        f.write(line)
    open_win(TRACKER)
    open_win(REPORTS)


if __name__ == "__main__":
    main()

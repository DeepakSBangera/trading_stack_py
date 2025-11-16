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
    artifacts = "; ".join(
        [
            "wk11_blend_targets.csv",
            "risk_schedule_blended.csv",
            "wk6_portfolio_compare.csv",
        ]
    )
    gates = "Targets = W6 capped base × (macro/DD × event); respect allow_new/rebalance flags"
    risks = "If many days have rebalancing blocked, drift may persist; monitor turnover"
    decisions = "Feed target_w into order sizing; keep cash for <1 gross if desired"

    line = f'{today},S-W11,~1.0h,"{artifacts}","{gates}","{risks}","{decisions}"\n'
    with TRACKER.open("a", encoding="utf-8") as f:
        f.write(line)
    open_win(TRACKER)
    open_win(REPORTS)


if __name__ == "__main__":
    main()

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
            "regime_timeline.csv",
            "macro_gates_eval.csv",
            "regime_risk_schedule.csv",
            "macro_gates.yaml",
        ]
    )
    gates = "Macro gate: BULL/NEUTRAL & vol<=threshold & DD<max; else FAIL"
    risks = "Proxy-based regimes; may lag turning points"
    decisions = "Apply total_risk_multiplier to order sizing; block promos if macro_gate=FAIL"
    line = f'{today},S-W7,~1.0h,"{artifacts}","{gates}","{risks}","{decisions}"\n'
    with TRACKER.open("a", encoding="utf-8") as f:
        f.write(line)
    open_win(TRACKER)
    open_win(REPORTS)


if __name__ == "__main__":
    main()

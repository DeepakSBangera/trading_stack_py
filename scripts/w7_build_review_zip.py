from __future__ import annotations

import datetime as dt
import os
import sys
import zipfile
from pathlib import Path

ROOT = Path(r"F:\Projects\trading_stack_py")
REPORTS = ROOT / "reports"
CONFIGS = [ROOT / "config" / "macro_gates.yaml", ROOT / "config" / "kill_switch.yaml"]

FILES = [
    REPORTS / "regime_timeline.csv",
    REPORTS / "macro_gates_eval.csv",
    REPORTS / "regime_risk_schedule.csv",
    REPORTS / "w7_validation_report.csv",
]


def open_win(p: Path):
    if sys.platform.startswith("win") and p.exists():
        os.startfile(p)


def main():
    ts = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_zip = REPORTS / f"W7_review_{ts}.zip"
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for f in FILES + [p for p in CONFIGS if p.exists()]:
            if f.exists():
                z.write(f, arcname=f.name)
    print("Created:", out_zip)
    open_win(out_zip)


if __name__ == "__main__":
    main()

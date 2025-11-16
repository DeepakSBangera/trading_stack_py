from __future__ import annotations

import datetime as dt
import os
import sys
import zipfile
from pathlib import Path

ROOT = Path(r"F:\Projects\trading_stack_py")
REPORTS = ROOT / "reports"

FILES = [
    REPORTS / "wk12_orders_schedule.csv",
    REPORTS / "wk12_orders_lastday.csv",
    REPORTS / "wk12_orders_validation.csv",
]


def open_win(p: Path):
    if sys.platform.startswith("win") and p.exists():
        os.startfile(p)


def main():
    ts = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_zip = REPORTS / f"W12_review_{ts}.zip"
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for f in FILES:
            if f.exists():
                z.write(f, arcname=f.name)
    print("Created:", out_zip)
    open_win(out_zip)


if __name__ == "__main__":
    main()

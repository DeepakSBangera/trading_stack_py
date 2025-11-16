from __future__ import annotations

import datetime as dt
import os
import sys
import zipfile
from pathlib import Path

ROOT = Path(r"F:\Projects\trading_stack_py")
REPORTS = ROOT / "reports"
CONFIG = ROOT / "config" / "capacity_policy.yaml"

FILES = [
    REPORTS / "turnover_profile.csv",
    REPORTS / "liquidity_screens.csv",
    REPORTS / "pretrade_violations.csv",
    REPORTS / "session_summary_w3.csv",
    REPORTS / "orders_daily_throttled.csv",
    REPORTS / "canary_log.csv",
    CONFIG,
]


def open_win(p: Path):
    if sys.platform.startswith("win") and p.exists():
        os.startfile(p)


def main():
    ts = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_zip = REPORTS / f"W3_review_{ts}.zip"
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for f in FILES:
            if f.exists():
                z.write(f, arcname=f.name)
    print("Created:", out_zip)
    # Open the reports folder highlighting the new zip
    open_win(out_zip)


if __name__ == "__main__":
    main()

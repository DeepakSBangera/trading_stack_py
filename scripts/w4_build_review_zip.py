# scripts/w4_build_review_zip.py
from __future__ import annotations

import datetime as dt
import json
import zipfile
from pathlib import Path

ROOT = Path(r"F:\Projects\trading_stack_py")
REPORTS = ROOT / "reports"
OUT_ZIP = REPORTS / f"W4_review_{dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.zip"

INCLUDE = [
    REPORTS / "wk4_voltarget_stops.csv",
    REPORTS / "dd_throttle_map.csv",
    REPORTS / "kill_switch.yaml",
    ROOT / "scripts" / "w4_voltarget_stops.py",
]


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in INCLUDE:
            if p.exists():
                z.write(p, arcname=p.relative_to(ROOT))
    print(json.dumps({"created": str(OUT_ZIP)}, indent=2))


if __name__ == "__main__":
    main()

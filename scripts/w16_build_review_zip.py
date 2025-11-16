from __future__ import annotations

import datetime as dt
import json
import zipfile
from pathlib import Path

ROOT = Path(r"F:\Projects\trading_stack_py")
REPORTS = ROOT / "reports"
OUT = REPORTS / f"W16_review_{dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.zip"

INCLUDE = [
    REPORTS / "w16_caps_property_tests.csv",
    REPORTS / "w16_caps_edge_cases.csv",
    REPORTS / "w16_caps_diag.json",
    ROOT / "scripts" / "w16_caps_tests.py",
]


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in INCLUDE:
            if p.exists():
                z.write(p, arcname=p.relative_to(ROOT))
    print(json.dumps({"created": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()

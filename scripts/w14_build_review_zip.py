# scripts/w14_build_review_zip.py
from __future__ import annotations

import datetime as dt
import json
import zipfile
from pathlib import Path

ROOT = Path(r"F:\Projects\trading_stack_py")
REPORTS = ROOT / "reports"
OUT = REPORTS / f"W14_review_{dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.zip"

INCLUDE = [
    REPORTS / "w14_positions_reco.csv",
    REPORTS / "w14_drift_summary.csv",
    REPORTS / "w14_reco_diag.json",
    ROOT / "scripts" / "w14_t1_reco.py",
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

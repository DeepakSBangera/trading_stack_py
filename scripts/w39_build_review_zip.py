from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def main():
    ts = pd.Timestamp.now(tz="Asia/Kolkata").strftime("%Y-%m-%d_%H-%M-%S")
    zip_path = REPORTS / f"W39_capacity_review_{ts}.zip"
    REPORTS.mkdir(parents=True, exist_ok=True)

    want = [
        REPORTS / "wk39_capacity_audit.csv",
        REPORTS / "wk39_capacity_summary.json",
    ]
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in want:
            if p.exists():
                z.write(p, arcname=p.name)

    print(json.dumps({"created": str(zip_path)}))


if __name__ == "__main__":
    main()

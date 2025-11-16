from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(r"F:\Projects\trading_stack_py")
REPORTS = ROOT / "reports"
DOCS = ROOT / "docs"

OUT = (
    REPORTS / f"W44_redteam_review_{pd.Timestamp.utcnow().tz_convert('Asia/Kolkata').strftime('%Y-%m-%d_%H-%M-%S')}.zip"
)

INCLUDE = [
    REPORTS / "wk44_redteam_summary.json",
    DOCS / "wk44_redteam_report.md",
    REPORTS / "W44_redteam_snapshot.zip",  # from drill
]


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in INCLUDE:
            if p.exists():
                z.write(p, arcname=p.name)
    print(json.dumps({"created": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()

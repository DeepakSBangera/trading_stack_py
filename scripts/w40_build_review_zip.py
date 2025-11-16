from __future__ import annotations

import json
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(r"F:\Projects\trading_stack_py")
REPORTS = ROOT / "reports"


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = REPORTS / f"W40_exec_review_{ts}.zip"
    wants = [
        REPORTS / "wk40_exec_quality.csv",
        REPORTS / "wk40_exec_quality_summary.json",
        ROOT / "docs" / "living_tracker.csv",
    ]
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in wants:
            if p.exists():
                zf.write(p, arcname=p.relative_to(ROOT))
    print(json.dumps({"created": str(out)}, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations
from pathlib import Path
from datetime import datetime
import zipfile
import json

ROOT = Path(r"F:\Projects\trading_stack_py")
REPORTS = ROOT / "reports"

def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = REPORTS / f"W39_capacity_review_{ts}.zip"
    wants = [
        REPORTS / "wk39_capacity_audit.csv",
        REPORTS / "wk39_capacity_summary.json",
        REPORTS / "wk39_capacity_stress.csv",
        ROOT / "docs" / "living_tracker.csv",
    ]
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in wants:
            if p.exists():
                zf.write(p, arcname=p.relative_to(ROOT))
    info = {"created": str(out)}
    print(json.dumps(info, indent=2))

if __name__ == "__main__":
    main()


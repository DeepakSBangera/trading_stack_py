from __future__ import annotations

import json
import time
import zipfile
from pathlib import Path

ROOT = Path(r"F:\Projects\trading_stack_py")
REPORTS = ROOT / "reports"
OUTS = [
    REPORTS / "wk43_barbell_compare.csv",
    REPORTS / "wk43_barbell_compare_summary.json",
    ROOT / "docs" / "living_tracker.csv",
]


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    z = REPORTS / f"W43_barbell_review_{ts}.zip"
    with zipfile.ZipFile(z, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in OUTS:
            if p.exists():
                zf.write(p, arcname=p.relative_to(ROOT))
    print(json.dumps({"created": str(z)}, indent=2))


if __name__ == "__main__":
    main()

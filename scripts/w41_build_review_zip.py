from __future__ import annotations

import json
import time
import zipfile
from pathlib import Path

ROOT = Path(r"F:\Projects\trading_stack_py")
REPORTS = ROOT / "reports"
OUTS = [
    REPORTS / "wk41_momentum_tilt.csv",
    REPORTS / "wk41_momentum_tilt_summary.json",
    ROOT / "docs" / "living_tracker.csv",
]


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    zpath = REPORTS / f"W41_momentum_review_{ts}.zip"
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in OUTS:
            if p.exists():
                zf.write(p, arcname=p.relative_to(ROOT))
    print(json.dumps({"created": str(zpath)}, indent=2))


if __name__ == "__main__":
    main()

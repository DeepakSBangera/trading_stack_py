from __future__ import annotations

import datetime as dt
import json
import zipfile
from pathlib import Path

ROOT = Path(r"F:\Projects\trading_stack_py")
REPORTS = ROOT / "reports"

FILES = [
    "wk34_leverage_throttle.csv",
    "w34_leverage_summary.json",
    "regime_timeline.csv",
    "wk11_blend_targets.csv",
    "dd_throttle_map.csv",
]


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = REPORTS / f"W34_review_{ts}.zip"
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name in FILES:
            p = REPORTS / name
            if p.exists():
                z.write(p, arcname=p.name)
    print(json.dumps({"created": str(out)}, indent=2))


if __name__ == "__main__":
    main()

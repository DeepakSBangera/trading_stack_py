from __future__ import annotations

import datetime as dt
import json
import zipfile
from pathlib import Path

ROOT = Path(r"F:\Projects\trading_stack_py")
REPORTS = ROOT / "reports"

FILES = [
    "wk38_turnover_profile.csv",
    "wk38_turnover_reco.json",
    "wk11_blend_targets.csv",
]


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = REPORTS / f"W38_turnover_review_{ts}.zip"
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name in FILES:
            p = REPORTS / name
            if p.exists():
                z.write(p, arcname=p.name)
    print(json.dumps({"created": str(out)}, indent=2))


if __name__ == "__main__":
    main()

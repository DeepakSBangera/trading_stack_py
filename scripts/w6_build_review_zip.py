# scripts/w6_build_review_zip.py
from __future__ import annotations

import datetime as dt
import json
import zipfile
from pathlib import Path

ROOT = Path(r"F:\Projects\trading_stack_py")
REPORTS = ROOT / "reports"
OUT = REPORTS / f"W6_review_{dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.zip"

INCLUDE = [
    REPORTS / "wk6_portfolio_compare.csv",
    REPORTS / "factor_exposure_weekly.csv",
    REPORTS / "capacity_curve.csv",
    REPORTS / "wk6_weights_capped.csv",
    REPORTS / "wk6_caps_validation.csv",
    ROOT / "scripts" / "w6_optimizer_compare.py",
    ROOT / "scripts" / "w6_enforce_caps.py",
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

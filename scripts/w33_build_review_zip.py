from __future__ import annotations

import datetime as dt
import json
import zipfile
from pathlib import Path

ROOT = Path(r"F:\Projects\trading_stack_py")
REPORTS = ROOT / "reports"

FILES = [
    "w33_barbell_results.csv",
    "w33_barbell_summary.json",
    "w33_barbell_sleeves_view.csv",
    "list2_conviction.csv",
    "list3_quality.csv",
]


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = REPORTS / f"W33_review_{ts}.zip"
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name in FILES:
            p = REPORTS / name
            if p.exists():
                z.write(p, arcname=p.name)
    print(json.dumps({"created": str(out)}, indent=2))


if __name__ == "__main__":
    main()

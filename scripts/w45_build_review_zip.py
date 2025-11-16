from __future__ import annotations

import json
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(r"F:\Projects\trading_stack_py")
REPORTS = ROOT / "reports"
DOCS = ROOT / "docs"

SUMMARY_JSON = REPORTS / "wk45_e2e_summary.json"


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_zip = REPORTS / f"W45_review_{ts}.zip"

    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in [
            SUMMARY_JSON,
            REPORTS / "run_manifest.jsonl",
            DOCS / "living_tracker.csv",
        ]:
            if p.exists():
                z.write(p, p.name)

        # include any W45 freeze zips created in this session
        for zf in REPORTS.glob("W45_freeze_*.zip"):
            z.write(zf, zf.name)

    payload = {"created": str(out_zip)}
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

# scripts/w19_build_review_zip.py
from __future__ import annotations

import datetime as dt
import json
import zipfile
from pathlib import Path

ROOT = Path(r"F:\Projects\trading_stack_py")
REPORTS = ROOT / "reports"
WIRES = REPORTS / "wires"


def main():
    ts = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = REPORTS / f"W19_wire_review_{ts}.zip"
    keep = [
        REPORTS / "wk12_orders_lastday.csv",
        REPORTS / "w18_pretrade_ok.csv",
        WIRES / "w19_orders_wire.csv",
        WIRES / "w19_wire_summary.json",
    ]
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in keep:
            if p.exists():
                z.write(p, p.relative_to(REPORTS))
    print(json.dumps({"created": str(out)}, indent=2))


if __name__ == "__main__":
    main()

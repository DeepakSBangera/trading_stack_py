from __future__ import annotations

import datetime as dt
import json
import zipfile
from pathlib import Path

ROOT = Path(r"F:\Projects\trading_stack_py")
REPORTS = ROOT / "reports"

FILES = [
    "w31_bandit_assignments.csv",
    "w31_exec_bandit_diag.json",
    "exec_bandit_log.csv",
    "wk13_dryrun_fills.csv",
    r"wires\w19_orders_wire.csv",
]


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = REPORTS / f"W31_review_{ts}.zip"
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name in FILES:
            p = REPORTS / name
            if "\\" in name and not p.exists():
                # handle nested (wires\..) explicitly
                p = REPORTS / name
            if p.exists():
                z.write(
                    p,
                    arcname=p.name if p.parent == REPORTS else name.replace("\\", "_"),
                )
    print(json.dumps({"created": str(out)}, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import datetime as dt
import json
import zipfile
from pathlib import Path

ROOT = Path(r"F:\Projects\trading_stack_py")
REPORTS = ROOT / "reports"


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = REPORTS / f"W26_review_{ts}.zip"
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in [
            REPORTS / "w26_daily_portfolio_returns.csv",
            REPORTS / "w26_stress_table.csv",
            REPORTS / "wk26_ops_cvar.csv",
            ROOT / "docs" / "living_tracker.csv",
        ]:
            if p.exists():
                z.write(p, arcname=p.name)
    print(json.dumps({"created": str(out)}, indent=2))


if __name__ == "__main__":
    main()

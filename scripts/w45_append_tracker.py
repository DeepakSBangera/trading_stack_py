from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(r"F:\Projects\trading_stack_py")
DOCS = ROOT / "docs"
TRACKER = DOCS / "living_tracker.csv"

SESSION = "S-W45"
NOTE = "W45: E2E dry-run + production freeze + review pack"


def main():
    DOCS.mkdir(parents=True, exist_ok=True)
    now_ist = pd.Timestamp.now(tz="Asia/Kolkata").isoformat()

    row = {"session": SESSION, "ts_ist": now_ist, "note": NOTE}

    if TRACKER.exists():
        try:
            df = pd.read_csv(TRACKER)
        except Exception:
            df = pd.DataFrame(columns=["session", "ts_ist", "note"])
    else:
        df = pd.DataFrame(columns=["session", "ts_ist", "note"])

    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(TRACKER, index=False, encoding="utf-8")

    print({"tracker_csv": str(TRACKER), "session": SESSION})


if __name__ == "__main__":
    main()

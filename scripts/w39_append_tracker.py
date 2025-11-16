from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
TRACKER = DOCS / "living_tracker.csv"


def main():
    DOCS.mkdir(parents=True, exist_ok=True)
    if TRACKER.exists():
        df = pd.read_csv(TRACKER)
    else:
        df = pd.DataFrame(columns=["session", "ts_ist", "note"])

    row = {
        "session": "S-W39",
        "ts_ist": pd.Timestamp.now(tz="Asia/Kolkata").isoformat(),
        "note": "Capacity & Liquidity Discipline — ADV caps, min ₹ADV, stress 2x/3x",
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(TRACKER, index=False)
    print({"tracker_csv": str(TRACKER), "session": "S-W39", "note": "n/a"})


if __name__ == "__main__":
    main()

from __future__ import annotations

from datetime import datetime
from pathlib import Path

ROOT = Path(r"F:\Projects\trading_stack_py")
DOCS = ROOT / "docs"
DOCS.mkdir(parents=True, exist_ok=True)
TRACKER = DOCS / "living_tracker.csv"


def main() -> None:
    if not TRACKER.exists():
        TRACKER.write_text("date,session,hours,artifacts,gates,risks,decisions\n", encoding="utf-8")
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    row = f'{now},S-W43,4,"reports/wk43_barbell_compare.csv; reports/wk43_barbell_compare_summary.json","W0 gates ok","Caps applied","Split {{"L1":0.60,"L2":0.25,"L3":0.15}}"\n'
    with TRACKER.open("a", encoding="utf-8") as f:
        f.write(row)
    print({"tracker_csv": str(TRACKER), "session": "S-W43"})


if __name__ == "__main__":
    main()

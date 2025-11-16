from __future__ import annotations

from datetime import datetime
from pathlib import Path

ROOT = Path(r"F:\Projects\trading_stack_py")
DOCS = ROOT / "docs"
DOCS.mkdir(parents=True, exist_ok=True)
TRACKER = DOCS / "living_tracker.csv"


def main() -> None:
    if not TRACKER.exists():
        TRACKER.write_text(
            "date,session,hours,artifacts,gates,risks,decisions\n", encoding="utf-8"
        )

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    row = f'{now},S-W42,4,"reports/wk42_after_tax_schedule.csv; reports/wk42_after_tax_schedule_summary.json","W0 gates ok","Tax schedule seeded","See summary"\n'
    with TRACKER.open("a", encoding="utf-8") as f:
        f.write(row)
    print({"tracker_csv": str(TRACKER), "session": "S-W42"})


if __name__ == "__main__":
    main()

import csv
import datetime as dt
from pathlib import Path

ROOT = Path(r"F:\Projects\trading_stack_py")
DOCS = ROOT / "docs"
REPORTS = ROOT / "reports"
DOCS.mkdir(parents=True, exist_ok=True)
TRACKER = DOCS / "living_tracker.csv"
row = [
    "S-W24",
    dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "w24_black_litterman_compare.py",
    "wk24_black_litterman_compare.csv",
    "ok",
]
with open(TRACKER, "a", newline="", encoding="utf-8") as f:
    csv.writer(f).writerow(row)
print({"tracker_csv": str(TRACKER), "session": "S-W24"})

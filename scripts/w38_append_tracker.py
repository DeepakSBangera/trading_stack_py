from __future__ import annotations

import csv
import datetime as dt
import json
from pathlib import Path

ROOT = Path(r"F:\Projects\trading_stack_py")
DOCS = ROOT / "docs"
REPORTS = ROOT / "reports"
TRACKER = DOCS / "living_tracker.csv"
RECO = REPORTS / "wk38_turnover_reco.json"


def main():
    DOCS.mkdir(parents=True, exist_ok=True)
    reco_txt = "n/a"
    if RECO.exists():
        j = json.loads(RECO.read_text(encoding="utf-8"))
        r = (j or {}).get("recommended", {})
        if r:
            reco_txt = f"{r.get('rebalance', '?')} / band {r.get('band_bps', '?')} bps; cost≈{r.get('cost_drag_bps', '?')} bps; corr={r.get('tracking_corr', '?')}"
    fields = [
        "ts",
        "session",
        "artifact",
        "as_of",
        "results_csv",
        "best_note",
        "git_sha8",
    ]
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    write_header = not TRACKER.exists()
    with TRACKER.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        w.writerow(
            {
                "ts": now,
                "session": "S-W38",
                "artifact": "W38 Turnover Optimization",
                "as_of": "",
                "results_csv": str(REPORTS / "wk38_turnover_profile.csv"),
                "best_note": reco_txt,
                "git_sha8": "",
            }
        )
    print({"tracker_csv": str(TRACKER), "session": "S-W38", "note": reco_txt})


if __name__ == "__main__":
    main()

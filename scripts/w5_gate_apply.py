from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(r"F:\Projects\trading_stack_py")
REPORTS = ROOT / "reports"
WF_CSV = REPORTS / "wk5_walkforward_dsr.csv"
OUT_CSV = REPORTS / "w5_promote_retire.csv"
CANARY = REPORTS / "canary_log.csv"


def open_win(p: Path):
    if sys.platform.startswith("win") and p.exists():
        os.startfile(p)


def main():
    if not WF_CSV.exists():
        raise SystemExit(f"Missing {WF_CSV}. Run w5_walkforward_dsr.py first.")

    wf = pd.read_csv(WF_CSV)
    if wf.empty:
        raise SystemExit("Walk-forward file is empty.")

    # Apply gate (restate for clarity even if already in the CSV)
    wf["promote"] = (wf["dsr_out"] > 0.5) & (wf["sr_out"] > 0)

    # Summarize promotions / retirements by window
    out = wf[
        [
            "train_start",
            "train_end",
            "test_start",
            "test_end",
            "sr_in",
            "sr_out",
            "dsr_in",
            "dsr_out",
            "promote",
        ]
    ].copy()

    # Materialize action column
    out["action"] = out["promote"].map({True: "PROMOTE", False: "REVIEW/RETIRE"})
    REPORTS.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    # Append simple canary line
    line = f"W5 gate apply: rows={out.shape[0]}, promote={int(out['promote'].sum())}\n"
    with CANARY.open("a", encoding="utf-8") as f:
        f.write(line)

    print(
        json.dumps(
            {
                "rows": int(out.shape[0]),
                "promote": int(out["promote"].sum()),
                "retire_or_review": int((~out["promote"]).sum()),
                "out_csv": str(OUT_CSV),
                "canary_log": str(CANARY),
            },
            indent=2,
        )
    )

    # Open outputs
    open_win(OUT_CSV)
    open_win(CANARY)


if __name__ == "__main__":
    main()

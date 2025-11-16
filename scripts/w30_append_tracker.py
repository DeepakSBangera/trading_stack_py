from __future__ import annotations

import csv
import datetime as dt
import json
from pathlib import Path

ROOT = Path(r"F:\Projects\trading_stack_py")
DOCS = ROOT / "docs"
REPORTS = ROOT / "reports"
TRACKER = DOCS / "living_tracker.csv"
DIAG = REPORTS / "w30_risk_sens_diag.json"
WEIGHTS = REPORTS / "w30_risk_sens_policy.csv"


def main():
    DOCS.mkdir(parents=True, exist_ok=True)
    note = ""
    if DIAG.exists():
        j = json.loads(DIAG.read_text(encoding="utf-8"))
        src = "ACCEPTED" if j.get("accepted") else "BASELINE"
        lam = j.get("chosen_lambda")
        gam = j.get("chosen_gamma")
        cand = j.get("candidate_metrics", {})
        base = j.get("baseline_metrics", {})
        note = f"Risk-sens {src}; λ={lam}, γ={gam}; Δmean_bps≈{round((cand.get('mean', 0) - base.get('mean', 0)) / 1e-4, 2)}; CVaR95(bps) {round(base.get('cvar95', 0) / 1e-4, 2)}→{round(cand.get('cvar95', 0) / 1e-4, 2)}"

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
                "session": "S-W30",
                "artifact": "W30 Risk-Sensitive Policy",
                "as_of": "",
                "results_csv": str(WEIGHTS) if WEIGHTS.exists() else "",
                "best_note": note,
                "git_sha8": "",
            }
        )
    print(json.dumps({"tracker_csv": str(TRACKER), "session": "S-W30"}, indent=2))


if __name__ == "__main__":
    main()

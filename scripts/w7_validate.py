from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(r"F:\Projects\trading_stack_py")
REPORTS = ROOT / "reports"
CFG_MG = ROOT / "config" / "macro_gates.yaml"
OUT = REPORTS / "w7_validation_report.csv"


def open_win(p: Path):
    if sys.platform.startswith("win") and p.exists():
        os.startfile(p)


def load_yaml(p: Path) -> dict:
    import yaml

    return yaml.safe_load(p.read_text(encoding="utf-8"))


def main():
    tl = REPORTS / "regime_timeline.csv"
    sched = REPORTS / "regime_risk_schedule.csv"
    if not tl.exists():
        raise SystemExit("Missing reports\\regime_timeline.csv — run w7_compute_regimes.py first.")
    if not sched.exists():
        raise SystemExit("Missing reports\\regime_risk_schedule.csv — run w7_apply_gates.py first.")

    mg = load_yaml(CFG_MG) if CFG_MG.exists() else {"policy": {}}
    base_mult = float(mg.get("policy", {}).get("base_risk_multiplier", 1.0))
    hi_mult = float(mg.get("policy", {}).get("high_vol_multiplier", 0.75))
    bear_mult = float(mg.get("policy", {}).get("bear_multiplier", 0.50))
    bull_mult = float(mg.get("policy", {}).get("bull_multiplier", 1.00))
    neutral_mult = float(mg.get("policy", {}).get("neutral_multiplier", 0.90))

    tl_df = pd.read_csv(tl, parse_dates=["date"])
    sc_df = pd.read_csv(sched, parse_dates=["date"])

    # Basic checks
    checks = []
    checks.append({"check": "timeline_nonempty", "value": tl_df.shape[0] > 0, "limit": True})
    checks.append({"check": "schedule_nonempty", "value": sc_df.shape[0] > 0, "limit": True})

    # Bounds on multipliers
    min_allowed = base_mult * bear_mult * min(hi_mult, 1.0)  # worst case: BEAR & HIGH_VOL
    max_allowed = base_mult * bull_mult
    mmin = float(sc_df["total_risk_multiplier"].min())
    mmax = float(sc_df["total_risk_multiplier"].max())
    checks.append(
        {
            "check": "multiplier_min_ok",
            "value": mmin >= round(min_allowed - 1e-9, 6),
            "limit": f">={min_allowed}",
        }
    )
    checks.append(
        {
            "check": "multiplier_max_ok",
            "value": mmax <= round(max_allowed + 1e-9, 6),
            "limit": f"<={max_allowed}",
        }
    )

    # Gate consistency: if macro_gate == FAIL then total_risk_multiplier should be <= base*bear or <= neutral*base (conservative)
    fail = tl_df.merge(sc_df[["date", "total_risk_multiplier"]], on="date", how="left")
    conservative_cap = base_mult * max(bear_mult, neutral_mult)
    fail["ok_when_fail"] = (fail["macro_gate"] != "FAIL") | (fail["total_risk_multiplier"] <= conservative_cap + 1e-9)
    checks.append(
        {
            "check": "fail_days_conservative",
            "value": bool(fail["ok_when_fail"].all()),
            "limit": f"<= {round(conservative_cap, 3)} when FAIL",
        }
    )

    rep = pd.DataFrame(checks)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rep.to_csv(OUT, index=False)

    print(
        json.dumps(
            {
                "rows_timeline": int(tl_df.shape[0]),
                "rows_schedule": int(sc_df.shape[0]),
                "total_mult_min_max": [round(mmin, 3), round(mmax, 3)],
                "min_allowed": round(min_allowed, 3),
                "max_allowed": round(max_allowed, 3),
                "report_csv": str(OUT),
            },
            indent=2,
        )
    )

    open_win(OUT)
    open_win(CFG_MG)
    open_win(REPORTS)


if __name__ == "__main__":
    main()

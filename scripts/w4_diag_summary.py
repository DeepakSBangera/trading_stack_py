# scripts/w4_diag_summary.py
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(r"F:\Projects\trading_stack_py")
REPORTS = ROOT / "reports"

VOLSTOPS = REPORTS / "wk4_voltarget_stops.csv"
THROTTLE = REPORTS / "dd_throttle_map.csv"
KILLSW = REPORTS / "kill_switch.yaml"
OUT_JSON = REPORTS / "w4_diag_summary.json"


def main():
    out = {}
    if VOLSTOPS.exists():
        v = pd.read_csv(VOLSTOPS)
        out["volstops_rows"] = int(v.shape[0])
        atr_cols = [c for c in v.columns if c.startswith("atr_")]
        if atr_cols:
            c = atr_cols[0]
            out["atr_missing"] = int(v[c].isna().sum())
            out["atr_present"] = int((~v[c].isna()).sum())
        out["tickers"] = len(v["ticker"].unique())
    else:
        out["volstops_rows"] = 0

    if THROTTLE.exists():
        t = pd.read_csv(THROTTLE)
        out["throttle_rows"] = int(t.shape[0])
        out["throttle_min_max"] = [
            float(t["risk_multiplier"].min()),
            float(t["risk_multiplier"].max()),
        ]
    else:
        out["throttle_rows"] = 0

    out["kill_switch_exists"] = KILLSW.exists()

    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

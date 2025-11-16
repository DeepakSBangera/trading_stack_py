from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(r"F:\Projects\trading_stack_py")
REPORTS = ROOT / "reports"

SCHED_MACRO = REPORTS / "regime_risk_schedule.csv"  # from W7
FLAGS_EVENT = REPORTS / "events_position_flags.csv"  # from W8
OUT = REPORTS / "risk_schedule_blended.csv"


def open_win(p: Path):
    if sys.platform.startswith("win") and p.exists():
        os.startfile(p)


def main():
    if not SCHED_MACRO.exists():
        raise SystemExit(
            "Missing reports\\regime_risk_schedule.csv — run w7_apply_gates.py first."
        )
    if not FLAGS_EVENT.exists():
        raise SystemExit(
            "Missing reports\\events_position_flags.csv — run w8_apply_event_rules.py first."
        )

    macro = pd.read_csv(SCHED_MACRO, parse_dates=["date"])
    evt = pd.read_csv(FLAGS_EVENT, parse_dates=["date"])

    # evt: date,ticker,allow_new,rebalance_allowed,risk_mult,event_note
    # macro: date, total_risk_multiplier, macro_gate, (trend_regime, vol_regime, etc.)
    cols_macro = [
        "date",
        "total_risk_multiplier",
        "macro_gate",
        "trend_regime",
        "vol_regime",
        "dd_pct",
    ]
    macro_slim = macro[cols_macro].copy()

    # Merge per day & ticker
    df = evt.merge(macro_slim, on="date", how="left")
    if df["total_risk_multiplier"].isna().any():
        # backfill in case dates misaligned (shouldn't happen with our synthetic data)
        df["total_risk_multiplier"] = df["total_risk_multiplier"].ffill().bfill()
        df["macro_gate"] = df["macro_gate"].ffill().bfill()
        df["trend_regime"] = df["trend_regime"].ffill().bfill()
        df["vol_regime"] = df["vol_regime"].ffill().bfill()
        df["dd_pct"] = df["dd_pct"].ffill().bfill()

    # Final permissions: if macro gate FAIL → no new positions and no rebalance
    gate_fail = df["macro_gate"] == "FAIL"
    df["allow_new_final"] = (~gate_fail) & df["allow_new"]
    df["rebalance_allowed_final"] = (~gate_fail) & df["rebalance_allowed"]

    # Final multiplier = macro/DD multiplier × event multiplier
    df["event_multiplier"] = df["risk_mult"].astype(float)
    df["macro_multiplier"] = df["total_risk_multiplier"].astype(float)
    df["final_risk_multiplier"] = (
        df["macro_multiplier"] * df["event_multiplier"]
    ).clip(lower=0.0)

    # Output tidy columns
    out = df[
        [
            "date",
            "ticker",
            "trend_regime",
            "vol_regime",
            "dd_pct",
            "macro_gate",
            "macro_multiplier",
            "event_multiplier",
            "final_risk_multiplier",
            "allow_new_final",
            "rebalance_allowed_final",
            "event_note",
        ]
    ].sort_values(["date", "ticker"])

    REPORTS.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    print(
        json.dumps(
            {
                "out_csv": str(OUT),
                "rows": int(out.shape[0]),
                "final_mult_min_max": [
                    float(out["final_risk_multiplier"].min()),
                    float(out["final_risk_multiplier"].max()),
                ],
                "blocked_days_examples": int((~out["allow_new_final"]).sum()),
            },
            indent=2,
        )
    )

    open_win(OUT)


if __name__ == "__main__":
    main()

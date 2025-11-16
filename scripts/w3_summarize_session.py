from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(r"F:\Projects\trading_stack_py")
REPORTS = ROOT / "reports"
OUT = REPORTS / "session_summary_w3.csv"
CFG = ROOT / "config" / "capacity_policy.yaml"


def open_win(p: Path):
    if sys.platform.startswith("win") and p.exists():
        os.startfile(p)


def safe_read_csv(p: Path, **kw):
    return pd.read_csv(p, **kw) if p.exists() else pd.DataFrame()


def main():
    turn = safe_read_csv(REPORTS / "turnover_profile.csv", parse_dates=["date"])
    liq = safe_read_csv(REPORTS / "liquidity_screens.csv", parse_dates=["date"])
    pv = safe_read_csv(REPORTS / "pretrade_violations.csv", parse_dates=["date"])

    if turn.empty or liq.empty or pv.empty:
        raise SystemExit("Missing artifacts; run w3_make_costs_and_churn.py first.")

    summary = {
        "median_daily_turnover_pct": round(float(turn["turnover_pct"].median()), 4),
        "p90_daily_turnover_pct": round(float(turn["turnover_pct"].quantile(0.90)), 4),
        "p95_daily_turnover_pct": round(float(turn["turnover_pct"].quantile(0.95)), 4),
        "sum_turnover_pct_over_window": round(float(turn["turnover_pct"].sum()), 4),
        "liquidity_rows": int(liq.shape[0]),
        "liquidity_violations": int(liq["violation"].sum()),
        "pretrade_rows": int(pv.shape[0]),
        "pretrade_violations": int(pv["violation"].sum()),
    }
    print(json.dumps(summary, indent=2))

    pd.DataFrame([summary]).to_csv(OUT, index=False)
    open_win(OUT)
    open_win(CFG)


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(r"F:\Projects\trading_stack_py")
REPORTS = ROOT / "reports"
CFG = ROOT / "config" / "capacity_policy.yaml"


def open_win(p: Path):
    if sys.platform.startswith("win") and p.exists():
        os.startfile(p)


def main():
    turn = pd.read_csv(REPORTS / "turnover_profile.csv", parse_dates=["date"])
    liq = pd.read_csv(REPORTS / "liquidity_screens.csv", parse_dates=["date"])
    pv = pd.read_csv(REPORTS / "pretrade_violations.csv", parse_dates=["date"])

    summ = {
        "turnover": {
            "days": int(turn.shape[0]),
            "median_daily_pct": float(turn["turnover_pct"].median()),
            "p90_daily_pct": float(turn["turnover_pct"].quantile(0.90)),
            "p95_daily_pct": float(turn["turnover_pct"].quantile(0.95)),
            "sum_daily_pct": float(turn["turnover_pct"].sum()),
        },
        "liquidity": {
            "rows": int(liq.shape[0]),
            "violations": int(liq["violation"].sum()),
            "violators_top5": liq[liq["violation"] == True]["ticker"]
            .value_counts()
            .head(5)
            .to_dict(),
        },
        "pretrade": {
            "rows": int(pv.shape[0]),
            "violations": int(pv["violation"].sum()),
            "violators_top5": pv[pv["violation"] == True]["ticker"]
            .value_counts()
            .head(5)
            .to_dict(),
        },
    }
    print(json.dumps(summ, indent=2))

    # Open your artifacts + policy
    open_win(REPORTS / "turnover_profile.csv")
    open_win(REPORTS / "liquidity_screens.csv")
    open_win(REPORTS / "pretrade_violations.csv")
    open_win(CFG)


if __name__ == "__main__":
    main()

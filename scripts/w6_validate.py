from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(r"F:\Projects\trading_stack_py")
REPORTS = ROOT / "reports"
CONFIG = ROOT / "config" / "capacity_policy.yaml"
OUT = REPORTS / "w6_validation_report.csv"

SECTOR_MAP = {
    "HDFCBANK.NS": "Financials",
    "RELIANCE.NS": "Energy",
    "TCS.NS": "IT",
    "INFY.NS": "IT",
    "AXISBANK.NS": "Financials",
    "ICICIBANK.NS": "Financials",
    "SBIN.NS": "Financials",
    "BHARTIARTL.NS": "Telecom",
    "ITC.NS": "Staples",
    "LT.NS": "Industrials",
    "KOTAKBANK.NS": "Financials",
    "BAJFINANCE.NS": "Financials",
    "HCLTECH.NS": "IT",
    "MARUTI.NS": "Discretionary",
    "SUNPHARMA.NS": "Healthcare",
    "TITAN.NS": "Discretionary",
    "ULTRACEMCO.NS": "Materials",
    "ONGC.NS": "Energy",
    "NESTLEIND.NS": "Staples",
    "ASIANPAINT.NS": "Materials",
}


def open_win(p: Path):
    if sys.platform.startswith("win") and p.exists():
        os.startfile(p)


def load_yaml(p: Path) -> dict:
    import yaml

    return yaml.safe_load(p.read_text(encoding="utf-8"))


def main():
    comp_csv = REPORTS / "wk6_portfolio_compare.csv"
    pos_pq = REPORTS / "positions_daily.parquet"
    if not comp_csv.exists():
        raise SystemExit("Missing reports\\wk6_portfolio_compare.csv — run w6_portfolio_compare.py first.")
    if not pos_pq.exists():
        raise SystemExit("Missing reports\\positions_daily.parquet — bootstrap first.")

    pol = load_yaml(CONFIG) if CONFIG.exists() else {}
    sector_cap = float(pol.get("sector_cap_base_pct", 35)) / 100.0
    per_name_cap = 0.12

    comp = pd.read_csv(comp_csv)
    # sanity
    s_w = comp["w_final"].sum()
    max_name = comp["w_final"].max()
    sec = comp.groupby("sector")["w_final"].sum().sort_values(ascending=False)
    sec_breach = sec > sector_cap
    name_breach = comp["w_final"] > per_name_cap

    report = pd.DataFrame(
        {
            "check": [
                "weights_sum_to_one",
                "max_name_weight",
                "any_name_breach",
                "any_sector_breach",
            ],
            "value": [
                round(float(s_w), 6),
                round(float(max_name), 6),
                bool(name_breach.any()),
                bool(sec_breach.any()),
            ],
            "limit": [1.0, per_name_cap, False, False],
        }
    )
    REPORTS.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUT, index=False)

    summary = {
        "weights_sum": round(float(s_w), 6),
        "max_name_final": round(float(max_name), 6),
        "per_name_cap": per_name_cap,
        "any_name_breach": bool(name_breach.any()),
        "sector_cap": sector_cap,
        "sector_maxes": {k: round(float(v), 6) for k, v in sec.to_dict().items()},
        "any_sector_breach": bool(sec_breach.any()),
        "out_csv": str(OUT),
    }
    print(json.dumps(summary, indent=2))
    open_win(OUT)
    open_win(CONFIG)


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(r"F:\Projects\trading_stack_py")
REPORTS = ROOT / "reports"
CONFIG = ROOT / "config" / "capacity_policy.yaml"
OUT = REPORTS / "capacity_curve.csv"


def open_win(p: Path):
    if sys.platform.startswith("win") and p.exists():
        os.startfile(p)


def load_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main():
    adv_pq = REPORTS / "adv_value.parquet"
    if not adv_pq.exists():
        raise SystemExit("Missing reports\\adv_value.parquet — run bootstrap first.")
    if not CONFIG.exists():
        raise SystemExit("Missing config\\capacity_policy.yaml — create it first.")

    pol = load_yaml(CONFIG)
    cap_pct = float(pol.get("adv_cap_pct_L1", pol.get("adv_cap_pct", 10))) / 100.0

    adv = pd.read_parquet(adv_pq)
    adv["date"] = pd.to_datetime(adv["date"])

    # per-name daily cap in currency + total capacity
    adv["per_name_cap_value"] = cap_pct * adv["adv_value"]
    daily = (
        adv.groupby("date", as_index=False)["per_name_cap_value"]
        .sum()
        .rename(columns={"per_name_cap_value": "total_capacity_value"})
    )

    # simple curve: percentiles of per-name caps (cross-section) each day
    pc = (
        adv.groupby("date")["per_name_cap_value"]
        .quantile([0.5, 0.75, 0.9, 0.95])
        .unstack()
    )
    pc.columns = [f"p{int(q * 100)}_per_name_cap" for q in pc.columns]
    curve = daily.merge(pc.reset_index(), on="date", how="left").sort_values("date")

    curve.to_csv(OUT, index=False)

    print(
        json.dumps(
            {
                "out_csv": str(OUT),
                "cap_pct_L1": cap_pct,
                "dates": int(curve.shape[0]),
                "total_capacity_min_max": [
                    float(curve["total_capacity_value"].min()),
                    float(curve["total_capacity_value"].max()),
                ],
            },
            indent=2,
        )
    )

    open_win(OUT)
    open_win(CONFIG)


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"F:\Projects\trading_stack_py")
REPORTS = ROOT / "reports"
CFG_KS = ROOT / "config" / "kill_switch.yaml"
OUT_CSV = REPORTS / "dd_throttle_map.csv"


def open_win(p: Path):
    if sys.platform.startswith("win") and p.exists():
        os.startfile(p)


def load_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main():
    ks = load_yaml(CFG_KS) if CFG_KS.exists() else {"policy": {}}
    throttle_floor = float(ks.get("policy", {}).get("throttle_floor", 0.25))

    # Drawdown buckets (% from peak)
    buckets = [
        (0, 5),
        (5, 10),
        (10, 15),
        (15, 20),
        (20, 25),
        (25, 30),
        (30, 35),
        (35, 40),
    ]
    # Risk multipliers descending to throttle_floor
    mults = np.linspace(1.00, throttle_floor, num=len(buckets))

    rows = []
    for (lo, hi), m in zip(buckets, mults, strict=False):
        rows.append(
            {
                "dd_from_peak_low_pct": lo,
                "dd_from_peak_high_pct": hi,
                "risk_throttle_multiplier": round(float(m), 3),
                "notes": f"Scale gross & order bands to {round(float(m) * 100, 1)}% when DD in [{lo},{hi})%",
            }
        )
    df = pd.DataFrame(rows)
    REPORTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print(
        json.dumps(
            {
                "out_csv": str(OUT_CSV),
                "buckets": df[
                    [
                        "dd_from_peak_low_pct",
                        "dd_from_peak_high_pct",
                        "risk_throttle_multiplier",
                    ]
                ].to_dict(orient="records"),
            },
            indent=2,
        )
    )

    open_win(OUT_CSV)
    open_win(CFG_KS)


if __name__ == "__main__":
    main()

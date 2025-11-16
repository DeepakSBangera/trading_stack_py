from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"F:\Projects\trading_stack_py")
REPORTS = ROOT / "reports"
CFG_MAIN = ROOT / "config" / "capacity_policy.yaml"
CFG_KS = ROOT / "config" / "kill_switch.yaml"
OUT_CSV = REPORTS / "wk4_voltarget_stops.csv"


def open_win(p: Path):
    if sys.platform.startswith("win") and p.exists():
        os.startfile(p)


def load_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def ann_from_daily_std(std_daily: float, days: int = 252) -> float:
    return float(std_daily) * math.sqrt(days) * 100.0  # to %


def main():
    pos_pq = REPORTS / "positions_daily.parquet"
    if not pos_pq.exists():
        raise SystemExit(f"Missing {pos_pq}. Run your bootstrap (W3) first.")

    # Inputs
    ks = load_yaml(CFG_KS) if CFG_KS.exists() else {"policy": {}}
    pol_main = load_yaml(CFG_MAIN) if CFG_MAIN.exists() else {}
    vol_target = float(
        ks.get("policy", {}).get(
            "vol_target_annual_pct", pol_main.get("vol_target_annual_pct", 12.0)
        )
    )
    kelly_cap = float(
        ks.get("policy", {}).get("kelly_cap", pol_main.get("kelly_base", 0.25))
    )

    # Build daily portfolio returns from positions' port_value
    pos = pd.read_parquet(pos_pq)
    port = (
        pos[["date", "port_value"]]
        .drop_duplicates()
        .sort_values("date")
        .assign(date=lambda d: pd.to_datetime(d["date"]))
        .set_index("date")
    )
    port["ret"] = port["port_value"].pct_change().fillna(0.0)

    # Realized statistics
    std_d = float(port["ret"].std(ddof=1))
    mu_d = float(port["ret"].mean())
    vol_ann_pct = ann_from_daily_std(std_d)
    mu_ann_pct = mu_d * 252 * 100.0

    # Scale to target: multiply gross by this factor (clip to [0.25, 2.0] for safety)
    scale_to_target = float(
        np.clip((vol_target / vol_ann_pct) if vol_ann_pct > 0 else 1.0, 0.25, 2.0)
    )

    # Kelly suggestion (approx): k = mu/var, clipped to [0, kelly_cap]
    var_d = float(port["ret"].var(ddof=1))
    kelly_raw = (mu_d / var_d) if var_d > 0 else 0.0
    kelly_suggest = float(np.clip(kelly_raw, 0.0, kelly_cap))

    # Stop % proxy: 20d rolling std * N (N=2.5 default)
    N = 2.5
    port["roll20_std"] = port["ret"].rolling(20).std().bfill().fillna(std_d)
    port["stop_pct"] = (port["roll20_std"] * N * 100.0).clip(0, 50)  # cap at 50%

    # Emit per-day row plus summary on last row
    out = port.reset_index()[["date", "ret", "roll20_std", "stop_pct"]].copy()
    out["realized_ann_vol_pct"] = vol_ann_pct
    out["target_ann_vol_pct"] = vol_target
    out["scale_to_target"] = scale_to_target
    out["kelly_suggest"] = kelly_suggest
    out["mu_ann_pct"] = mu_ann_pct

    REPORTS.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    print(
        json.dumps(
            {
                "vol_ann_pct": round(vol_ann_pct, 4),
                "mu_ann_pct": round(mu_ann_pct, 4),
                "scale_to_target": round(scale_to_target, 4),
                "kelly_suggest": round(kelly_suggest, 4),
                "out_csv": str(OUT_CSV),
            },
            indent=2,
        )
    )

    open_win(OUT_CSV)
    open_win(CFG_KS)


if __name__ == "__main__":
    main()

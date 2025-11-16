from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(r"F:\Projects\trading_stack_py")
REPORTS = ROOT / "reports"
CFG_MG = ROOT / "config" / "macro_gates.yaml"
CFG_KS = ROOT / "config" / "kill_switch.yaml"
OUT = REPORTS / "regime_risk_schedule.csv"


def open_win(p: Path):
    if sys.platform.startswith("win") and p.exists():
        os.startfile(p)


def load_yaml(p: Path) -> dict:
    import yaml

    return yaml.safe_load(p.read_text(encoding="utf-8"))


def bucketize_dd(dd_pct: float, buckets):
    for lo, hi, m in buckets:
        if lo <= dd_pct < hi:
            return m
    return buckets[-1][2]


def main():
    tl = REPORTS / "regime_timeline.csv"
    throttle_csv = REPORTS / "dd_throttle_map.csv"
    if not tl.exists():
        raise SystemExit(
            "Missing regime_timeline.csv — run w7_compute_regimes.py first."
        )
    if not throttle_csv.exists():
        raise SystemExit("Missing dd_throttle_map.csv — run W4 dd_throttle_map first.")

    mg = load_yaml(CFG_MG) if CFG_MG.exists() else {"policy": {}}
    ks = load_yaml(CFG_KS) if CFG_KS.exists() else {"policy": {}}
    base_mult = float(mg.get("policy", {}).get("base_risk_multiplier", 1.0))
    m_highvol = float(mg.get("policy", {}).get("high_vol_multiplier", 0.75))
    m_bear = float(mg.get("policy", {}).get("bear_multiplier", 0.50))
    m_bull = float(mg.get("policy", {}).get("bull_multiplier", 1.00))
    m_neutral = float(mg.get("policy", {}).get("neutral_multiplier", 0.90))

    tl_df = pd.read_csv(tl, parse_dates=["date"])
    th = pd.read_csv(throttle_csv)

    # Build DD throttle buckets list: [(lo, hi, mult), ...]
    buckets = []
    # infer multiplier column
    mult_col = "risk_throttle_multiplier"
    for _, r in th.iterrows():
        lo = float(r["dd_from_peak_low_pct"])
        hi = float(r["dd_from_peak_high_pct"])
        m = float(r[mult_col])
        buckets.append((lo, hi, m))

    rows = []
    for _, r in tl_df.iterrows():
        dd_pct = abs(float(r["from_peak_dd_pct"]))
        dd_mult = bucketize_dd(dd_pct, buckets)
        # regime multiplier
        regime = str(r["trend_regime"])
        vol = str(r["vol_regime"])
        reg_mult = (
            m_bull if regime == "BULL" else m_bear if regime == "BEAR" else m_neutral
        )
        if vol == "HIGH_VOL":
            reg_mult *= m_highvol / 1.0  # additional scaling for high vol

        total_mult = base_mult * dd_mult * reg_mult
        rows.append(
            {
                "date": r["date"],
                "trend_regime": regime,
                "vol_regime": vol,
                "dd_pct": float(r["from_peak_dd_pct"]),
                "dd_multiplier": round(dd_mult, 3),
                "regime_multiplier": round(reg_mult, 3),
                "base_multiplier": round(base_mult, 3),
                "total_risk_multiplier": round(total_mult, 3),
                "macro_gate": str(r["macro_gate"]),
            }
        )
    out = pd.DataFrame(rows).sort_values("date")
    out.to_csv(OUT, index=False)

    print(
        json.dumps(
            {
                "out_csv": str(OUT),
                "dates": int(out.shape[0]),
                "total_mult_min_max": [
                    float(out["total_risk_multiplier"].min()),
                    float(out["total_risk_multiplier"].max()),
                ],
            },
            indent=2,
        )
    )

    open_win(OUT)
    open_win(CFG_MG)


if __name__ == "__main__":
    main()

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
CFG_MG = ROOT / "config" / "macro_gates.yaml"
OUT_TIMELINE = REPORTS / "regime_timeline.csv"
OUT_EVAL = REPORTS / "macro_gates_eval.csv"


def open_win(p: Path):
    if sys.platform.startswith("win") and p.exists():
        os.startfile(p)


def load_yaml(p: Path) -> dict:
    import yaml

    return yaml.safe_load(p.read_text(encoding="utf-8"))


def ann_vol_from_daily(std_daily: float) -> float:
    return float(std_daily) * (252**0.5) * 100.0


def main():
    pos_pq = REPORTS / "positions_daily.parquet"
    if not pos_pq.exists():
        raise SystemExit(
            "Missing reports\\positions_daily.parquet — run W3 bootstrap first."
        )
    pos = pd.read_parquet(pos_pq)
    port = (
        pos[["date", "port_value"]]
        .drop_duplicates()
        .sort_values("date")
        .assign(date=lambda d: pd.to_datetime(d["date"]))
        .set_index("date")
    )
    port["ret"] = port["port_value"].pct_change().fillna(0.0)

    mg = load_yaml(CFG_MG) if CFG_MG.exists() else {"inputs": {}, "policy": {}}
    ks = load_yaml(CFG_KS) if CFG_KS.exists() else {"policy": {}}

    trend_n = int(mg.get("inputs", {}).get("trend_lookback_days", 60))
    vol_n = int(mg.get("inputs", {}).get("vol_window_days", 20))
    vol_mult = float(mg.get("inputs", {}).get("vol_high_mult_vs_target", 1.5))
    dd_lvls = list(mg.get("inputs", {}).get("dd_levels_pct", [5, 10, 15, 20, 25, 30]))

    tgt_vol = float(
        ks.get("policy", {}).get(
            "vol_target_annual_pct",
            mg.get("policy", {}).get("default_target_ann_vol_pct", 12.0),
        )
    )
    allow_bear = bool(mg.get("policy", {}).get("allow_trade_in_bear", False))
    allow_highvol = bool(mg.get("policy", {}).get("allow_trade_in_high_vol", True))
    max_trailing = float(
        mg.get("policy", {}).get(
            "max_trailing_dd_pct", ks.get("policy", {}).get("max_trailing_dd_pct", 30.0)
        )
    )

    base_mult = float(mg.get("policy", {}).get("base_risk_multiplier", 1.0))
    m_highvol = float(mg.get("policy", {}).get("high_vol_multiplier", 0.75))
    m_bear = float(mg.get("policy", {}).get("bear_multiplier", 0.50))
    m_bull = float(mg.get("policy", {}).get("bull_multiplier", 1.00))
    m_neutral = float(mg.get("policy", {}).get("neutral_multiplier", 0.90))

    # Compute rolling stats
    ser = port["port_value"].copy()
    ma = ser.rolling(trend_n, min_periods=1).mean()
    trend_regime = pd.Series(
        np.where(
            ser > ma * 1.005, "BULL", np.where(ser < ma * 0.995, "BEAR", "NEUTRAL")
        ),
        index=ser.index,
    )

    roll_std = port["ret"].rolling(vol_n, min_periods=1).std().bfill()
    roll_vol_ann = roll_std.apply(ann_vol_from_daily)
    vol_regime = pd.Series(
        np.where(roll_vol_ann >= vol_mult * tgt_vol, "HIGH_VOL", "NORMAL_VOL"),
        index=ser.index,
    )

    # Trailing drawdown (% from peak on port_value)
    peak = ser.cummax()
    dd = (ser / peak - 1.0) * 100.0  # negative numbers
    dd_abs = dd.abs()  # positive %
    dd_regime = pd.cut(
        dd_abs,
        bins=[-1] + dd_lvls + [1000],
        labels=[
            f"{a}-{b}%"
            for a, b in zip(
                [0] + dd_lvls, dd_lvls + [">" + str(dd_lvls[-1])], strict=False
            )
        ],
        include_lowest=True,
    )

    # Overall gate
    gate_fail = (
        ((trend_regime == "BEAR") & (not allow_bear))
        | ((vol_regime == "HIGH_VOL") & (not allow_highvol))
        | (dd_abs >= max_trailing)
    )
    gate = pd.Series(np.where(gate_fail, "FAIL", "PASS"), index=ser.index)

    timeline = pd.DataFrame(
        {
            "date": ser.index,
            "port_value": ser.values,
            "ma_trend": ma.values,
            "trend_regime": trend_regime.values,
            "roll20_ann_vol_pct": roll_vol_ann.values,
            "vol_regime": vol_regime.values,
            "from_peak_dd_pct": dd.values,
            "dd_bucket": dd_regime.astype(str).values,
            "macro_gate": gate.values,
        }
    ).sort_values("date")

    REPORTS.mkdir(parents=True, exist_ok=True)
    timeline.to_csv(OUT_TIMELINE, index=False)

    # Last-day evaluation summary
    last = timeline.iloc[-1]
    eval_row = pd.DataFrame(
        [
            {
                "as_of": str(last["date"]),
                "trend_regime": last["trend_regime"],
                "vol_regime": last["vol_regime"],
                "from_peak_dd_pct": float(last["from_peak_dd_pct"]),
                "macro_gate": last["macro_gate"],
                "target_ann_vol_pct": tgt_vol,
                "vol_high_threshold_pct": vol_mult * tgt_vol,
                "allow_trade_in_bear": allow_bear,
                "allow_trade_in_high_vol": allow_highvol,
                "max_trailing_dd_pct": max_trailing,
            }
        ]
    )
    eval_row.to_csv(OUT_EVAL, index=False)

    print(
        json.dumps(
            {
                "out_timeline": str(OUT_TIMELINE),
                "out_eval": str(OUT_EVAL),
                "as_of": str(last["date"]),
                "trend": last["trend_regime"],
                "vol": last["vol_regime"],
                "dd_pct": round(float(last["from_peak_dd_pct"]), 3),
                "gate": last["macro_gate"],
            },
            indent=2,
        )
    )

    open_win(OUT_TIMELINE)
    open_win(OUT_EVAL)
    open_win(CFG_MG)


if __name__ == "__main__":
    main()

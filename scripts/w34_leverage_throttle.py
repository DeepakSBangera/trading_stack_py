from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

"""
W34 — Safe Leverage Switch (1.10–1.25×) with Auto-Revert

Logic (policy defaults; tune safely):
- Base leverage = 1.00×
- Consider switching ON to 1.10–1.25× only when ALL are true on a given date:
  1) Macro gate PASS (from W7) — regime_timeline.gate == "PASS"
  2) Drawdown small  — regime_timeline.dd_pct ≥ -0.05  (>-5%)
  3) Trend favorable — regime_timeline.trend in {"UP","BULL","POSITIVE"} (case-insensitive)
  4) No DD throttle active today (optional; if dd_throttle_map.csv present and value<1.0 → OFF)

- When ON, pick leverage by signal quality proxy:
    1.25× if trend is strongly favorable (trend in {"UP","BULL"} and vol in {"LOW_VOL","NORMAL_VOL"})
    1.15× otherwise (mildly favorable)
  You can adjust this to a momentum metric later.

- Auto-revert to 1.00× if any guard fails on any day.

Inputs
- reports/regime_timeline.csv   (date, trend, vol, dd_pct, gate)
- reports/wk11_blend_targets.csv (date coverage)
- optional: reports/dd_throttle_map.csv (date, throttle ∈ (0,1]; if <1 => throttle ON)

Outputs
- reports/wk34_leverage_throttle.csv
    columns: date, gate, trend, vol, dd_pct, dd_throttle, leverage, reason
- reports/wk34_leverage_summary.json
    includes counts and policy knobs
"""

ROOT = Path(r"F:\Projects\trading_stack_py")
REPORTS = ROOT / "reports"

REGIME_CSV = REPORTS / "regime_timeline.csv"
CORE_TARGETS = REPORTS / "wk11_blend_targets.csv"
DD_THROTTLE = REPORTS / "dd_throttle_map.csv"

OUT_CSV = REPORTS / "wk34_leverage_throttle.csv"
OUT_SUMMARY = REPORTS / "wk34_leverage_summary.json"

# --- Policy knobs (safe defaults) ---
BASE_LEV = 1.00
LEV_HI = 1.25
LEV_LO = 1.10

DD_OK_THRESHOLD = -0.05  # ≥ -5% allowed
FAVORABLE_TRENDS = {"UP", "BULL", "POSITIVE"}
FAVORABLE_VOL = {"LOW_VOL", "NORMAL_VOL"}  # else treat as neutral/hostile


def _read_csv(p: Path) -> pd.DataFrame | None:
    try:
        if p.exists():
            return pd.read_csv(p)
    except Exception:
        pass
    return None


def _clean_regime(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    # expected columns: date, trend, vol, dd_pct, gate
    cols = {c.lower(): c for c in d.columns}
    d["date"] = pd.to_datetime(d[cols.get("date", "date")], errors="coerce").dt.date
    d["trend"] = d[cols.get("trend", "trend")].astype(str).str.upper().str.strip()
    d["vol"] = d[cols.get("vol", "vol")].astype(str).str.upper().str.strip()
    d["gate"] = d[cols.get("gate", "gate")].astype(str).str.upper().str.strip()
    d["dd_pct"] = pd.to_numeric(d[cols.get("dd_pct", "dd_pct")], errors="coerce")
    d = d.dropna(subset=["date"])
    return d[["date", "trend", "vol", "dd_pct", "gate"]]


def _load_dates_from_core() -> list:
    core = _read_csv(CORE_TARGETS)
    if core is None or core.empty or "date" not in core.columns:
        return []
    dates = (
        pd.to_datetime(core["date"], errors="coerce").dt.date.dropna().unique().tolist()
    )
    return sorted(dates)


def _load_throttle() -> dict:
    m = {}
    df = _read_csv(DD_THROTTLE)
    if df is None or df.empty:
        return m
    cols = {c.lower(): c for c in df.columns}
    dcol = cols.get("date", "date")
    tcol = cols.get("throttle", "throttle")
    if dcol not in df.columns or tcol not in df.columns:
        return m
    tmp = df.copy()
    tmp[dcol] = pd.to_datetime(tmp[dcol], errors="coerce").dt.date
    tmp[tcol] = pd.to_numeric(tmp[tcol], errors="coerce")
    tmp = tmp.dropna(subset=[dcol])
    for _, r in tmp.iterrows():
        m[r[dcol]] = float(r[tcol]) if math.isfinite(r[tcol]) else 1.0
    return m


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)

    regime = _read_csv(REGIME_CSV)
    if regime is None or regime.empty:
        OUT_CSV.write_text("", encoding="utf-8")
        OUT_SUMMARY.write_text(
            json.dumps(
                {"rows": 0, "note": f"Missing or empty {REGIME_CSV}; run W7 first."},
                indent=2,
            ),
            encoding="utf-8",
        )
        print(json.dumps({"rows": 0, "note": "no_regime"}, indent=2))
        return

    regime = _clean_regime(regime)
    dates_core = _load_dates_from_core()
    if not dates_core:
        # fall back to regime dates
        dates = sorted(regime["date"].unique().tolist())
    else:
        # align to core dates window
        dates = dates_core

    throttle_map = _load_throttle()

    rows = []
    for d in dates:
        rr = regime[regime["date"] == d]
        if rr.empty:
            # default OFF if no info
            rows.append(
                {
                    "date": d,
                    "gate": "NA",
                    "trend": "NA",
                    "vol": "NA",
                    "dd_pct": np.nan,
                    "dd_throttle": np.nan,
                    "leverage": BASE_LEV,
                    "reason": "no_regime_info",
                }
            )
            continue

        r = rr.iloc[0]
        gate = str(r["gate"])
        trend = str(r["trend"])
        vol = str(r["vol"])
        dd = float(r["dd_pct"]) if math.isfinite(r["dd_pct"]) else np.nan
        thr = throttle_map.get(d, 1.0)

        # Guards
        gate_ok = gate == "PASS"
        dd_ok = (dd >= DD_OK_THRESHOLD) if math.isfinite(dd) else False
        trend_ok = trend in FAVORABLE_TRENDS
        throttle_ok = thr >= 1.0  # if throttle < 1 → risk throttled → do not lever up

        if gate_ok and dd_ok and trend_ok and throttle_ok:
            # Choose level
            if vol in FAVORABLE_VOL:
                lev = LEV_HI
                reason = "favorable: gate+dd+trend+vol"
            else:
                lev = LEV_LO
                reason = "favorable: gate+dd+trend (vol not low)"
        else:
            lev = BASE_LEV
            reason_bits = []
            if not gate_ok:
                reason_bits.append("gate")
            if not dd_ok:
                reason_bits.append("dd")
            if not trend_ok:
                reason_bits.append("trend")
            if not throttle_ok:
                reason_bits.append("dd_throttle")
            reason = (
                "auto_revert: " + ",".join(reason_bits)
                if reason_bits
                else "auto_revert: unknown"
            )

        rows.append(
            {
                "date": d,
                "gate": gate,
                "trend": trend,
                "vol": vol,
                "dd_pct": dd,
                "dd_throttle": thr if math.isfinite(thr) else np.nan,
                "leverage": lev,
                "reason": reason,
            }
        )

    out = pd.DataFrame(rows)
    out = out.sort_values("date")
    out.to_csv(OUT_CSV, index=False)

    summary = {
        "rows": int(out.shape[0]),
        "on_days": int((out["leverage"] > BASE_LEV).sum()),
        "off_days": int((out["leverage"] <= BASE_LEV).sum()),
        "policy": {
            "BASE_LEV": BASE_LEV,
            "LEV_LO": LEV_LO,
            "LEV_HI": LEV_HI,
            "DD_OK_THRESHOLD": DD_OK_THRESHOLD,
            "FAVORABLE_TRENDS": sorted(list(FAVORABLE_TRENDS)),
            "FAVORABLE_VOL": sorted(list(FAVORABLE_VOL)),
        },
        "files": {
            "regime_timeline": str(REGIME_CSV),
            "wk11_targets": str(CORE_TARGETS),
            "dd_throttle_map": str(DD_THROTTLE),
            "out_csv": str(OUT_CSV),
        },
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "leverage_csv": str(OUT_CSV),
                "rows": summary["rows"],
                "on_days": summary["on_days"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

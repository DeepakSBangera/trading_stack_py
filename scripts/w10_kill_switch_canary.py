# scripts/w10_kill_switch_canary.py
from __future__ import annotations

import csv
import datetime as dt
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(r"F:\Projects\trading_stack_py")
REPORTS = ROOT / "reports"

KS_YAML = REPORTS / "kill_switch.yaml"
THROTTLE_FALLBACK = REPORTS / "dd_throttle_map.csv"
OUT_CANARY_CSV = REPORTS / "kill_switch_canary.csv"
OUT_CANARY_LAST = REPORTS / "kill_switch_canary_last.json"


# simple YAML-ish parser (no external deps)
def _parse_yaml_scalars(text: str) -> dict:
    out = {}
    for ln in text.splitlines():
        if ln.strip().startswith("#") or ":" not in ln:
            continue
        key, val = ln.split(":", 1)
        key = key.strip()
        val = val.strip()
        if val.endswith("#"):
            val = val.split("#", 1)[0].strip()
        # try float/int/bool
        low = val.lower()
        if low in ("true", "false"):
            out[key] = low == "true"
            continue
        try:
            if "." in val or "e" in low:
                out[key] = float(val)
            else:
                out[key] = int(val)
            continue
        except Exception:
            pass
        out[key] = val
    return out


def _extract_path_for_throttle(yaml_text: str) -> Path | None:
    # look for: throttle_map_csv: path
    m = re.search(r"throttle_map_csv\s*:\s*(.+)$", yaml_text, re.IGNORECASE | re.MULTILINE)
    if not m:
        return None
    val = m.group(1).strip()
    val = val.strip('"').strip("'")
    p = (ROOT / val) if not Path(val).is_absolute() else Path(val)
    return p


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().astimezone().isoformat(timespec="seconds")

    exists_yaml = KS_YAML.exists()
    ks_text = KS_YAML.read_text(encoding="utf-8") if exists_yaml else ""
    kv = _parse_yaml_scalars(ks_text) if exists_yaml else {}

    # validate core knobs (tolerant bounds)
    kelly = kv.get("kelly_base", 0.25)
    targ_vol = kv.get("target_vol_ann", 0.12)
    kelly_ok = isinstance(kelly, (int, float)) and 0.0 <= float(kelly) <= 1.0
    targ_ok = isinstance(targ_vol, (int, float)) and 0.02 <= float(targ_vol) <= 0.6

    # throttle table
    tpath = _extract_path_for_throttle(ks_text) if exists_yaml else None
    if tpath is None:
        tpath = THROTTLE_FALLBACK
    throttle_ok = None
    throttle_reason = ""
    if tpath.exists():
        try:
            t = pd.read_csv(tpath)
            # basic monotonicity: as dd_bucket_pct decreases (worse), risk_multiplier should not increase
            if {"dd_bucket_pct", "risk_multiplier"}.issubset(t.columns):
                t2 = t.sort_values("dd_bucket_pct")  # ascending: most positive to most negative
                # compute diffs (should be non-increasing as dd gets worse)
                dif = t2["risk_multiplier"].diff().fillna(0.0)
                bad = (dif > 0).sum()
                throttle_ok = bad == 0
                if not throttle_ok:
                    throttle_reason = f"non-monotone risk_multiplier in {bad} places"
            else:
                throttle_ok = False
                throttle_reason = "missing required columns in throttle map"
        except Exception as e:
            throttle_ok = False
            throttle_reason = f"read_error: {e}"
    else:
        throttle_ok = False
        throttle_reason = "throttle map file missing"

    # consolidate
    checks = [
        ("exists_yaml", exists_yaml, "" if exists_yaml else "kill_switch.yaml missing"),
        (
            "kelly_range",
            kelly_ok,
            "" if kelly_ok else f"kelly_base={kelly} out of [0,1]",
        ),
        (
            "target_vol_range",
            targ_ok,
            "" if targ_ok else f"target_vol_ann={targ_vol} out of [0.02,0.6]",
        ),
        ("throttle_table", throttle_ok, throttle_reason),
    ]
    pass_all = all(ok for _, ok, _ in checks)

    row = {
        "timestamp": ts,
        "pass": pass_all,
        "exists_yaml": exists_yaml,
        "kelly_ok": bool(kelly_ok),
        "target_vol_ok": bool(targ_ok),
        "throttle_ok": bool(throttle_ok),
        "reason_any": "; ".join([r for _, ok, r in checks if not ok and r]),
    }

    # append CSV
    write_header = not OUT_CANARY_CSV.exists()
    with OUT_CANARY_CSV.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)

    OUT_CANARY_LAST.write_text(json.dumps(row, indent=2), encoding="utf-8")
    print(json.dumps({"canary_csv": str(OUT_CANARY_CSV), **row}, indent=2))


if __name__ == "__main__":
    main()

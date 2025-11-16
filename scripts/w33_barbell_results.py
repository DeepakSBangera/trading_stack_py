from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

"""
W33 — Barbell: Core (List-1) + Conviction (List-2) + Long-Term/Quality (List-3)

Inputs
- Core/Systematic weights by date (from W11 blend): reports/wk11_blend_targets.csv
  required cols: ["date","ticker","target_w"]  (if base_w exists, it's ignored here)
- Optional sleeves (static per file; you can update anytime):
  reports/list2_conviction.csv  (cols: ["ticker","weight"])
  reports/list3_quality.csv     (cols: ["ticker","weight"])

Policy (defaults; tunable)
- Capital split: 60% (L1 Core) / 25% (L2 Conviction) / 15% (L3 LT/Quality)
- Per-name cap: 6% (0.06)
- Total names hard cap is handled elsewhere; here we just combine and cap.

Outputs
- reports/w33_barbell_results.csv  (date,ticker,w_core,w_list2,w_list3,w_total,caps_applied)
- reports/w33_barbell_summary.json (coverage, split, caps)
- reports/w33_barbell_sleeves_view.csv (expanded sleeve weights per date)

Notes
- If a sleeve file is empty/missing, that sleeve contributes 0 and result is just Core with split reweighted to keep totals at 100% (unless KEEP_SPLIT_STRICT=True).
"""

ROOT = Path(r"F:\Projects\trading_stack_py")
REPORTS = ROOT / "reports"

CORE_CSV = REPORTS / "wk11_blend_targets.csv"
L2_FILE = REPORTS / "list2_conviction.csv"
L3_FILE = REPORTS / "list3_quality.csv"

OUT_RESULTS = REPORTS / "w33_barbell_results.csv"
OUT_SUMMARY = REPORTS / "w33_barbell_summary.json"
OUT_SLEEVES = REPORTS / "w33_barbell_sleeves_view.csv"

# --- knobs ---
SPLIT_CORE = 0.60
SPLIT_L2 = 0.25
SPLIT_L3 = 0.15
PER_NAME_CAP = 0.06
KEEP_SPLIT_STRICT = False  # if a sleeve is empty, keep the split (leaves cash) vs renormalize active sleeves


def _read_csv(p: Path) -> pd.DataFrame | None:
    try:
        if p.exists():
            return pd.read_csv(p)
    except Exception:
        pass
    return None


def _z(v):
    return 0.0 if (v is None or not math.isfinite(v)) else float(v)


def _norm_weights(df: pd.DataFrame, wcol: str) -> pd.DataFrame:
    df = df.copy()
    x = pd.to_numeric(df[wcol], errors="coerce").fillna(0.0).clip(lower=0.0)
    s = float(x.sum())
    if s > 0:
        df[wcol] = x / s
    else:
        df[wcol] = 0.0
    return df


def _prepare_sleeve(file: Path, tag: str) -> pd.DataFrame:
    df = _read_csv(file)
    if df is None or df.empty or "ticker" not in df.columns:
        return pd.DataFrame(columns=["ticker", "weight", "sleeve"]).astype(
            {"ticker": str, "weight": float, "sleeve": str}
        )
    cols = {c.lower(): c for c in df.columns}
    tcol = cols.get("ticker", "ticker")
    wcol = cols.get("weight", "weight")
    out = df[[tcol, wcol]].rename(columns={tcol: "ticker", wcol: "weight"}).dropna()
    out["ticker"] = out["ticker"].astype(str)
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce").fillna(0.0).clip(lower=0.0)
    out = _norm_weights(out, "weight")
    out["sleeve"] = tag
    return out


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)

    core = _read_csv(CORE_CSV)
    if core is None or core.empty or not all(c in core.columns for c in ["date", "ticker", "target_w"]):
        OUT_RESULTS.write_text("", encoding="utf-8")
        OUT_SUMMARY.write_text(
            json.dumps(
                {
                    "rows": 0,
                    "note": f"Missing or empty {CORE_CSV}. Run W11 to produce wk11_blend_targets.csv.",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(json.dumps({"rows": 0, "note": "no_core"}, indent=2))
        return

    core = core.copy()
    core["date"] = pd.to_datetime(core["date"], errors="coerce").dt.date
    core["ticker"] = core["ticker"].astype(str)
    core["target_w"] = pd.to_numeric(core["target_w"], errors="coerce").fillna(0.0).clip(lower=0.0)

    # Normalize core per-date just in case (wk11 usually already normalized)
    core = core.groupby("date", as_index=False).apply(lambda d: _norm_weights(d, "target_w")).reset_index(drop=True)

    # Load sleeves
    l2 = _prepare_sleeve(L2_FILE, "L2")
    l3 = _prepare_sleeve(L3_FILE, "L3")

    has_l2 = not l2.empty
    has_l3 = not l3.empty

    # Decide effective splits
    s_core, s_l2, s_l3 = SPLIT_CORE, SPLIT_L2, SPLIT_L3
    if not KEEP_SPLIT_STRICT:
        active = []
        if core is not None and not core.empty:
            active.append("L1")
        if has_l2:
            active.append("L2")
        if has_l3:
            active.append("L3")
        if len(active) > 0:
            # renormalize only among active sleeves
            parts = []
            if "L1" in active:
                parts.append(SPLIT_CORE)
            if "L2" in active:
                parts.append(SPLIT_L2)
            if "L3" in active:
                parts.append(SPLIT_L3)
            tot = sum(parts)
            if tot > 0:
                f = 1.0 / tot
                if "L1" in active:
                    s_core = SPLIT_CORE * f
                else:
                    s_core = 0.0
                if "L2" in active:
                    s_l2 = SPLIT_L2 * f
                else:
                    s_l2 = 0.0
                if "L3" in active:
                    s_l3 = SPLIT_L3 * f
                else:
                    s_l3 = 0.0

    # Build per-date sleeves table
    dates = sorted(core["date"].unique().tolist())
    rows = []
    sleeves_rows = []

    # Pre-normalize sleeves (static) to 1.0 inside each sleeve
    if has_l2:
        l2 = _norm_weights(l2, "weight")
    if has_l3:
        l3 = _norm_weights(l3, "weight")

    for d in dates:
        cday = core[core["date"] == d][["ticker", "target_w"]].rename(columns={"target_w": "w_core"}).copy()
        # Apply core split
        cday["w_core"] = cday["w_core"] * s_core

        # Sleeve expansions (same set every date)
        if has_l2:
            l2_day = l2.copy()
            l2_day["w_list2"] = l2_day["weight"] * s_l2
        else:
            l2_day = pd.DataFrame(columns=["ticker", "w_list2"])

        if has_l3:
            l3_day = l3.copy()
            l3_day["w_list3"] = l3_day["weight"] * s_l3
        else:
            l3_day = pd.DataFrame(columns=["ticker", "w_list3"])

        # Merge all tickers
        merged = cday.merge(l2_day[["ticker", "w_list2"]], on="ticker", how="outer")
        merged = merged.merge(l3_day[["ticker", "w_list3"]], on="ticker", how="outer")
        merged = merged.fillna(0.0)

        merged["w_total_raw"] = merged["w_core"] + merged["w_list2"] + merged["w_list3"]

        # Per-name cap
        merged["w_total_capped"] = merged["w_total_raw"].clip(upper=PER_NAME_CAP)
        caps_applied = bool((merged["w_total_capped"] < merged["w_total_raw"]).any())

        # Renormalize to 1.0 if total < 1 due to caps? Keep total at <= 1 by default;
        # Many allocators prefer not to auto-renorm after caps; we'll keep totals as-is.
        # You can switch to renorm here if you want:
        total_alloc = float(merged["w_total_capped"].sum())
        # Optionally: if total_alloc > 0: merged["w_total_capped"] /= total_alloc

        merged.insert(0, "date", d)
        merged["caps_applied"] = caps_applied
        rows.append(
            merged[
                [
                    "date",
                    "ticker",
                    "w_core",
                    "w_list2",
                    "w_list3",
                    "w_total_capped",
                    "caps_applied",
                ]
            ]
        )

        # sleeves view
        if has_l2:
            l2_tmp = l2_day.copy()
            l2_tmp["date"] = d
            l2_tmp["sleeve"] = "L2"
            l2_tmp = l2_tmp[["date", "sleeve", "ticker", "w_list2"]]
            sleeves_rows.append(l2_tmp)
        if has_l3:
            l3_tmp = l3_day.copy()
            l3_tmp["date"] = d
            l3_tmp["sleeve"] = "L3"
            l3_tmp = l3_tmp[["date", "sleeve", "ticker", "w_list3"]]
            sleeves_rows.append(l3_tmp)

    out = (
        pd.concat(rows, ignore_index=True)
        if rows
        else pd.DataFrame(
            columns=[
                "date",
                "ticker",
                "w_core",
                "w_list2",
                "w_list3",
                "w_total_capped",
                "caps_applied",
            ]
        )
    )
    out = out.rename(columns={"w_total_capped": "w_total"})

    sleeves_view = (
        pd.concat(sleeves_rows, ignore_index=True)
        if sleeves_rows
        else pd.DataFrame(columns=["date", "sleeve", "ticker", "weight"])
    )
    if not sleeves_view.empty:
        sleeves_view = sleeves_view.rename(columns={"w_list2": "weight", "w_list3": "weight"})

    # Save
    out.to_csv(OUT_RESULTS, index=False)
    sleeves_view.to_csv(OUT_SLEEVES, index=False)

    summary = {
        "rows": int(out.shape[0]),
        "dates": len(dates),
        "tickers_unique": int(out["ticker"].nunique()) if not out.empty else 0,
        "splits_effective": {
            "core": round(s_core, 3),
            "list2": round(s_l2, 3),
            "list3": round(s_l3, 3),
        },
        "per_name_cap": PER_NAME_CAP,
        "keep_split_strict": KEEP_SPLIT_STRICT,
        "files": {
            "core": str(CORE_CSV),
            "list2": str(L2_FILE),
            "list3": str(L3_FILE),
            "results_csv": str(OUT_RESULTS),
            "sleeves_view_csv": str(OUT_SLEEVES),
        },
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"results_csv": str(OUT_RESULTS), "rows": summary["rows"]}, indent=2))


if __name__ == "__main__":
    main()

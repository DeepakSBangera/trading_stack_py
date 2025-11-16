# scripts/w16_caps_tests.py
from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path(r"F:\Projects\trading_stack_py")
REPORTS = ROOT / "reports"
DOCS = ROOT / "docs"
DATA = ROOT / "data" / "prices"

# Inputs (best-effort, optional sector map)
TARGETS_CSV = REPORTS / "wk11_blend_targets.csv"  # from W11
SECTORS_CSV_OPT = REPORTS / "sectors_map.csv"  # optional: ticker,sector
KILL_SWITCH_YAML = REPORTS / "kill_switch.yaml"  # optional: read caps if present

# Outputs
PROP_TESTS_CSV = REPORTS / "w16_caps_property_tests.csv"
EDGE_CASES_CSV = REPORTS / "w16_caps_edge_cases.csv"
DIAG_JSON = REPORTS / "w16_caps_diag.json"

# Defaults (fallbacks if not in kill_switch.yaml)
DEFAULT_PER_NAME_CAP = 0.10  # 10%
DEFAULT_SECTOR_CAP = 0.35  # 35% portfolio per sector

DATE_CANDS = ["date", "dt", "trading_day", "asof", "as_of"]
TICK_CANDS = ["ticker", "symbol", "name"]
WEIGHT_CANDS = ["target_w", "weight", "w", "blend_w", "final_w"]


def _pick(cols, cands):
    low = {c.lower(): c for c in cols}
    for k in cands:
        if k in low:
            return low[k]
    for c in cols:
        lc = c.lower().replace(" ", "").replace("-", "_")
        for k in cands:
            if lc == k.replace(" ", "").replace("-", "_"):
                return c
    return None


def _read_yaml_scalars(p: Path) -> dict:
    if not p.exists():
        return {}
    text = p.read_text(encoding="utf-8", errors="ignore")
    out = {}
    for ln in text.splitlines():
        if ":" not in ln or ln.strip().startswith("#"):
            continue
        k, v = ln.split(":", 1)
        k = k.strip()
        v = v.strip().split("#", 1)[0].strip()
        # try parse numbers/bool else keep string
        low = v.lower()
        if low in ("true", "false"):
            out[k] = low == "true"
            continue
        try:
            if any(ch in v for ch in ".eE"):
                out[k] = float(v)
            else:
                out[k] = int(v)
            continue
        except Exception:
            pass
        out[k] = v
    return out


def _caps_from_yaml() -> tuple[float, float]:
    kv = _read_yaml_scalars(KILL_SWITCH_YAML)
    # loose keys accepted
    per_name = None
    sector = None
    # Try several common keys
    for k in list(kv.keys()):
        lk = k.lower().replace(" ", "_")
        v = kv[k]
        if per_name is None and ("per_name_cap" in lk or "name_cap" in lk or "single_name_cap" in lk):
            try:
                per_name = float(v)
            except:
                pass
        if sector is None and ("sector_cap" in lk or "sector_caps_base" in lk or "sector_caps" in lk):
            try:
                sector = float(v)
            except:
                pass
    if per_name is None:
        per_name = DEFAULT_PER_NAME_CAP
    if sector is None:
        sector = DEFAULT_SECTOR_CAP
    # reasonable clamps
    per_name = float(min(max(per_name, 0.01), 1.0))
    sector = float(min(max(sector, 0.05), 1.0))
    return per_name, sector


def _load_targets_lastday() -> tuple[dt.date, pd.DataFrame]:
    if not TARGETS_CSV.exists():
        raise SystemExit(f"Missing {TARGETS_CSV}. Run W11 first.")
    df = pd.read_csv(TARGETS_CSV)
    d = _pick(df.columns, DATE_CANDS)
    t = _pick(df.columns, TICK_CANDS)
    w = _pick(df.columns, WEIGHT_CANDS)
    if not (d and t and w):
        raise SystemExit("wk11_blend_targets.csv missing date/ticker/weight-like columns")
    df[d] = pd.to_datetime(df[d], errors="coerce").dt.date
    last = df[d].max()
    sub = df[df[d] == last].copy()
    sub = sub[[t, w]].rename(columns={t: "ticker", w: "target_w"})
    sub["ticker"] = sub["ticker"].astype(str)
    sub["target_w"] = pd.to_numeric(sub["target_w"], errors="coerce").fillna(0.0).clip(lower=0.0)
    # renormalize just in case numerical noise
    s = sub["target_w"].sum()
    if s > 0:
        sub["target_w"] /= s
    return last, sub


def _load_sectors(universe: list[str]) -> pd.DataFrame:
    if SECTORS_CSV_OPT.exists():
        try:
            m = pd.read_csv(SECTORS_CSV_OPT)
            t = _pick(m.columns, TICK_CANDS)
            s = _pick(m.columns, ["sector", "industry", "gics", "name"])
            if t and s:
                sm = m[[t, s]].copy()
                sm.columns = ["ticker", "sector"]
                sm["ticker"] = sm["ticker"].astype(str)
                sm["sector"] = sm["sector"].astype(str)
                return sm
        except Exception:
            pass
    # Fallback: assign UNKNOWN
    return pd.DataFrame({"ticker": universe, "sector": ["UNKNOWN"] * len(universe)}, dtype=object)


def _property_tests(
    targets: pd.DataFrame, sectors: pd.DataFrame, per_name_cap: float, sector_cap: float
) -> pd.DataFrame:
    df = targets.merge(sectors, on="ticker", how="left")
    df["sector"] = df["sector"].fillna("UNKNOWN").astype(str)

    # per-name breaches
    df["per_name_breach"] = df["target_w"] > per_name_cap

    # sector sums & breaches
    sec = df.groupby("sector", as_index=False)["target_w"].sum().rename(columns={"target_w": "sector_sum"})
    sec["sector_breach"] = sec["sector_sum"] > sector_cap

    # attach sector sums to each row for reporting
    df = df.merge(sec, on="sector", how="left")

    # portfolio sums checks
    total_w = float(df["target_w"].sum())
    name_breaches = int(df["per_name_breach"].sum())
    sector_breaches = int(sec["sector_breach"].sum())

    rows = []
    for _, r in df.sort_values(["per_name_breach", "sector_breach"], ascending=False).iterrows():
        rows.append(
            {
                "ticker": r["ticker"],
                "sector": r["sector"],
                "target_w": round(float(r["target_w"]), 8),
                "sector_sum": (round(float(r["sector_sum"]), 8) if not math.isnan(float(r["sector_sum"])) else None),
                "per_name_cap": per_name_cap,
                "sector_cap": sector_cap,
                "per_name_breach": bool(r["per_name_breach"]),
                "sector_breach": bool(r["sector_sum"] > sector_cap if r["sector_sum"] == r["sector_sum"] else False),
            }
        )
    out = pd.DataFrame(rows)
    # portfolio summary row
    out.attrs["summary"] = {
        "names": int(len(df)),
        "sectors": int(df["sector"].nunique()),
        "total_w": total_w,
        "per_name_cap": per_name_cap,
        "sector_cap": sector_cap,
        "per_name_breaches": name_breaches,
        "sector_breaches": sector_breaches,
    }
    return out


def _edge_cases(targets: pd.DataFrame, sectors: pd.DataFrame, per_name_cap: float, sector_cap: float) -> pd.DataFrame:
    """
    Generate synthetic stress cases to ensure guards trip:
      - EC1: Force the largest name to exceed per-name cap (1.5x cap)
      - EC2: Force the largest sector to exceed sector cap (+10% to that sector, renorm others)
    """
    df = targets.merge(sectors, on="ticker", how="left")
    df["sector"] = df["sector"].fillna("UNKNOWN").astype(str)

    # EC1: bump the max name to 1.5x per-name cap (if cap>0)
    ec1 = df.copy()
    if per_name_cap > 0 and not ec1.empty:
        idx = ec1["target_w"].idxmax()
        ec1.loc[idx, "target_w"] = min(1.0, per_name_cap * 1.5)
        # renormalize others proportionally
        s = ec1["target_w"].sum()
        if s > 0:
            ec1["target_w"] /= s
    ec1_breach = bool((ec1["target_w"] > per_name_cap).any())

    # EC2: bump largest sector by +10% weight mass then renorm
    ec2 = df.copy()
    sec_sum = ec2.groupby("sector", as_index=False)["target_w"].sum().sort_values("target_w", ascending=False)
    if not sec_sum.empty:
        top_sec = sec_sum.iloc[0]["sector"]
        mask = ec2["sector"] == top_sec
        bump = 0.10  # add 10% mass to top sector, then renormalize
        ec2.loc[mask, "target_w"] = ec2.loc[mask, "target_w"] * (1.0 + bump)
        s = ec2["target_w"].sum()
        if s > 0:
            ec2["target_w"] /= s
    ec2_sec = ec2.groupby("sector", as_index=False)["target_w"].sum()
    ec2_breach = bool((ec2_sec["target_w"] > sector_cap).any())

    # assemble report
    rows = []
    rows.append(
        {
            "edge_case": "EC1_per_name_force",
            "breach_expected": True,
            "breach_observed": ec1_breach,
            "cap": per_name_cap,
        }
    )
    rows.append(
        {
            "edge_case": "EC2_sector_force",
            "breach_expected": True,
            "breach_observed": ec2_breach,
            "cap": sector_cap,
        }
    )
    return pd.DataFrame(rows)


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    last_day, targets = _load_targets_lastday()
    per_name_cap, sector_cap = _caps_from_yaml()

    # sectors
    sectors = _load_sectors(targets["ticker"].tolist())

    # property tests on true targets
    prop = _property_tests(targets, sectors, per_name_cap, sector_cap)
    sum_meta = prop.attrs.get("summary", {})
    prop.to_csv(PROP_TESTS_CSV, index=False)

    # edge cases to verify guards would trip
    edge = _edge_cases(targets, sectors, per_name_cap, sector_cap)
    edge.to_csv(EDGE_CASES_CSV, index=False)

    diag = {
        "as_of": str(last_day),
        "names": int(sum_meta.get("names", 0)),
        "sectors": int(sum_meta.get("sectors", 0)),
        "total_w": float(sum_meta.get("total_w", 0.0)),
        "per_name_cap": float(sum_meta.get("per_name_cap", DEFAULT_PER_NAME_CAP)),
        "sector_cap": float(sum_meta.get("sector_cap", DEFAULT_SECTOR_CAP)),
        "per_name_breaches": int(sum_meta.get("per_name_breaches", 0)),
        "sector_breaches": int(sum_meta.get("sector_breaches", 0)),
        "prop_tests_csv": str(PROP_TESTS_CSV),
        "edge_cases_csv": str(EDGE_CASES_CSV),
        "caps_source": (str(KILL_SWITCH_YAML) if KILL_SWITCH_YAML.exists() else "defaults"),
        "sectors_source": (str(SECTORS_CSV_OPT) if SECTORS_CSV_OPT.exists() else "fallback:UNKNOWN"),
    }
    DIAG_JSON.write_text(json.dumps(diag, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "prop_tests_csv": str(PROP_TESTS_CSV),
                "edge_cases_csv": str(EDGE_CASES_CSV),
                "per_name_breaches": diag["per_name_breaches"],
                "sector_breaches": diag["sector_breaches"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

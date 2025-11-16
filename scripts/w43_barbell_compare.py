from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

# ----------------- paths -----------------
ROOT = Path(r"F:\Projects\trading_stack_py")
DATA = ROOT / "data" / "csv"
DOCS = ROOT / "docs"
REPORTS = ROOT / "reports"

OUT_CSV = REPORTS / "wk43_barbell_compare.csv"
OUT_SUM = REPORTS / "wk43_barbell_compare_summary.json"

# Optional inputs this script will look for (first hit wins)
L1_CANDIDATES = [
    REPORTS / "wk11_alpha_blend.csv",  # preferred (columns: date,ticker,target_w or w)
    REPORTS / "wk41_momentum_tilt.csv",  # fallback (ticker,w or weight)
    REPORTS / "wk6_portfolio_compare.csv",  # another fallback
]

# Optional conviction & long-term lists (user-curated)
L2_FILE = (
    DOCS / "list2_conviction.csv"
)  # expected columns: ticker, target_w (or weight)
L3_FILE = DOCS / "list3_longterm.csv"  # expected columns: ticker, target_w (or weight)

# Optional sector map to enforce a light sector cap
SECTOR_MAP = DATA / "sectors_map.csv"  # columns: ticker, sector

# Policy knobs (can be adjusted)
SPLIT = {"L1_core": 0.60, "L2_conv": 0.25, "L3_long": 0.15}  # sums to 1.0
PER_NAME_CAP = 0.08  # 8% max per name after total blend
SECTOR_CAP = 0.35  # 35% sector cap (gross) if sector map is present
KEEP_TOP_N = 30  # hard 30-name cap


# ----------------- helpers -----------------
def _pick(cols, want) -> str | None:
    low = {c.lower(): c for c in cols}
    for w in want:
        if w.lower() in low:
            return low[w.lower()]
    return None


def _load_list1() -> pd.DataFrame:
    """Return df with columns: ticker, w_l1 (weights sum to 1 across L1 universe)."""
    for p in L1_CANDIDATES:
        if not p.exists():
            continue
        df = pd.read_csv(p)
        cols = df.columns
        tk = _pick(cols, ["ticker", "symbol", "name"])
        if tk is None:
            continue
        # many of our files carry date; we just want latest
        if "date" in cols:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"]).sort_values("date")
            if not df.empty:
                latest = df["date"].max()
                df = df[df["date"] == latest].copy()
        wt = _pick(cols, ["target_w", "w", "weight"])
        if wt:
            out = df[[tk, wt]].copy().rename(columns={tk: "ticker", wt: "w_l1"})
            out = out.groupby("ticker", as_index=False)["w_l1"].sum()
            s = float(out["w_l1"].abs().sum())
            if s > 0:
                out["w_l1"] = out["w_l1"] / s
            return out.sort_values("w_l1", ascending=False).reset_index(drop=True)
    # Synthetic fallback if nothing exists
    synth = pd.DataFrame(
        {
            "ticker": [f"SYN{i:02d}.NS" for i in range(1, 41)],
            "w_l1": np.linspace(1, 0.1, 40),
        }
    )
    synth["w_l1"] = synth["w_l1"] / synth["w_l1"].sum()
    return synth


def _load_user_list(path: Path, col_name: str) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    tk = _pick(df.columns, ["ticker", "symbol", "name"])
    wt = _pick(df.columns, ["target_w", "w", "weight"])
    if tk is None:
        return None
    out = df[[tk] + ([wt] if wt else [])].copy().rename(columns={tk: "ticker"})
    if wt:
        out.rename(columns={wt: col_name}, inplace=True)
        s = float(out[col_name].abs().sum())
        if s > 0:
            out[col_name] = out[col_name] / s
    else:
        out[col_name] = 1.0 / float(len(out))
    out = out.groupby("ticker", as_index=False)[col_name].sum()
    return out


def _merge_three_buckets(
    l1: pd.DataFrame, l2: pd.DataFrame | None, l3: pd.DataFrame | None
) -> pd.DataFrame:
    df = l1.copy()
    if "w_l1" not in df.columns:
        df["w_l1"] = 0.0
    df["w_l2"] = 0.0
    df["w_l3"] = 0.0
    if l2 is not None:
        df = df.merge(l2.rename(columns={"w_l2": "w_l2"}), on="ticker", how="outer")
    if l3 is not None:
        df = df.merge(l3.rename(columns={"w_l3": "w_l3"}), on="ticker", how="outer")
    df = df.fillna({"w_l1": 0.0, "w_l2": 0.0, "w_l3": 0.0})
    return df


def _apply_split(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize each bucket, then scale by SPLIT
    for col in ["w_l1", "w_l2", "w_l3"]:
        s = float(df[col].abs().sum())
        if s > 0:
            df[col] = df[col] / s
    df["w_core"] = df["w_l1"] * SPLIT["L1_core"]
    df["w_L2"] = df["w_l2"] * SPLIT["L2_conv"]
    df["w_L3"] = df["w_l3"] * SPLIT["L3_long"]
    df["w_total_pre"] = df["w_core"] + df["w_L2"] + df["w_L3"]
    # sort by total weight and keep top N
    df = df.sort_values("w_total_pre", ascending=False).head(KEEP_TOP_N).copy()
    # renormalize gross to 1.0
    s = float(df["w_total_pre"].abs().sum())
    if s > 0:
        df["w_total_pre"] = df["w_total_pre"] / s
    return df.reset_index(drop=True)


def _cap_per_name(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # ensure column exists even if something went empty
    if "w_total_pre" not in df.columns:
        df["w_total_pre"] = 0.0
    df["w_capped"] = df["w_total_pre"].clip(upper=PER_NAME_CAP)
    # renorm gross to 1 after cap (avoid div by zero)
    s = float(df["w_capped"].abs().sum())
    if s > 0:
        df["w_capped"] = df["w_capped"] / s
    return df


def _load_sector_map() -> Dict[str, str] | None:
    if not SECTOR_MAP.exists():
        return None
    m = pd.read_csv(SECTOR_MAP)
    tk = _pick(m.columns, ["ticker", "symbol", "name"])
    sc = _pick(m.columns, ["sector", "industry", "group"])
    if tk and sc:
        return dict(zip(m[tk].astype(str), m[sc].astype(str), strict=False))
    return None


def _cap_sector(df: pd.DataFrame, sec_map: Dict[str, str] | None) -> pd.DataFrame:
    df = df.copy()
    if sec_map is None:
        # Keep BOTH columns: w_capped (post name-cap) and w_total (final)
        df["sector"] = "NA"
        df["w_total"] = df["w_capped"]
        # final renorm to 1 just in case
        s = float(df["w_total"].sum())
        if s > 0:
            df["w_total"] = df["w_total"] / s
        return df.reset_index(drop=True)

    # Map sectors and cap
    df["sector"] = df["ticker"].astype(str).map(sec_map).fillna("NA")

    def _sector_scale(g: pd.DataFrame) -> pd.DataFrame:
        gross = float(g["w_capped"].sum())
        if gross <= SECTOR_CAP + 1e-12:
            g["w_total"] = g["w_capped"]
            return g
        scale = SECTOR_CAP / gross
        g["w_total"] = g["w_capped"] * scale
        return g

    tmp = df.groupby("sector", group_keys=False).apply(_sector_scale)
    # final renorm gross to 1
    s = float(tmp["w_total"].sum())
    if s > 0:
        tmp["w_total"] = tmp["w_total"] / s
    return tmp.reset_index(drop=True)


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)

    # Load L1
    l1 = _load_list1()

    # Load L2/L3 if present
    l2 = _load_user_list(L2_FILE, "w_l2")
    l3 = _load_user_list(L3_FILE, "w_l3")

    merged = _merge_three_buckets(l1, l2, l3)
    splitd = _apply_split(merged)
    capped = _cap_per_name(splitd)

    sec_map = _load_sector_map()
    final = _cap_sector(capped, sec_map)

    # Tidy output
    cols = [
        "ticker",
        "sector",
        "w_core",
        "w_L2",
        "w_L3",
        "w_total_pre",
        "w_capped",
        "w_total",
    ]
    # Some columns like w_core/L2/L3 may be missing if the bucket was empty; guard-select
    cols_exist = [c for c in cols if c in final.columns]
    out = (
        final[cols_exist]
        .copy()
        .sort_values("w_total", ascending=False if "w_total" in final.columns else True)
    )

    out.to_csv(OUT_CSV, index=False)

    # Roll-up summary
    k = {
        "as_of_ist": pd.Timestamp.utcnow().tz_convert("Asia/Kolkata").isoformat(),
        "names": int(out.shape[0]),
        "split": SPLIT,
        "per_name_cap": PER_NAME_CAP,
        "sector_cap": SECTOR_CAP if sec_map is not None else None,
        "files": {"detail_csv": str(OUT_CSV)},
        "inputs": {
            "l1_source": next(
                (str(p) for p in L1_CANDIDATES if p.exists()), "synthetic"
            ),
            "l2_present": bool(l2 is not None),
            "l3_present": bool(l3 is not None),
            "sector_map_present": bool(sec_map is not None),
        },
        "notes": "Barbell = L1(core) + L2(conviction) + L3(long-term); keep 30; per-name & optional sector caps.",
    }
    with open(OUT_SUM, "w", encoding="utf-8") as f:
        json.dump(k, f, indent=2, ensure_ascii=False)

    print(json.dumps(k, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

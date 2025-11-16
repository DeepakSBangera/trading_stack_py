from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(r"F:\Projects\trading_stack_py")
REPORTS = ROOT / "reports"
CONFIG = ROOT / "config" / "capacity_policy.yaml"
OUT = REPORTS / "wk6_portfolio_compare.csv"

SECTOR_MAP = {
    "HDFCBANK.NS": "Financials",
    "RELIANCE.NS": "Energy",
    "TCS.NS": "IT",
    "INFY.NS": "IT",
    "AXISBANK.NS": "Financials",
    "ICICIBANK.NS": "Financials",
    "SBIN.NS": "Financials",
    "BHARTIARTL.NS": "Telecom",
    "ITC.NS": "Staples",
    "LT.NS": "Industrials",
    "KOTAKBANK.NS": "Financials",
    "BAJFINANCE.NS": "Financials",
    "HCLTECH.NS": "IT",
    "MARUTI.NS": "Discretionary",
    "SUNPHARMA.NS": "Healthcare",
    "TITAN.NS": "Discretionary",
    "ULTRACEMCO.NS": "Materials",
    "ONGC.NS": "Energy",
    "NESTLEIND.NS": "Staples",
    "ASIANPAINT.NS": "Materials",
}


def open_win(p: Path):
    if sys.platform.startswith("win") and p.exists():
        os.startfile(p)


def load_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def cap_by_name(w: pd.Series, cap: float) -> pd.Series:
    w = w.clip(lower=0.0)
    if w.sum() == 0:
        return w
    w = w / w.sum()
    w = w.clip(upper=cap)
    if w.sum() == 0:
        return w
    return w / w.sum()


def enforce_sector_caps(
    w: pd.Series, sectors: pd.Series, sector_cap: float, iters: int = 20
) -> pd.Series:
    w = w.clip(lower=0.0)
    if w.sum() == 0:
        return w
    w = w / w.sum()
    for _ in range(iters):
        df = pd.DataFrame({"w": w, "sector": sectors})
        sec = df.groupby("sector")["w"].sum()
        viol = sec[sec > sector_cap]
        if viol.empty:
            break
        for s in viol.index:
            idx = df.index[df["sector"] == s]
            # scale down sector weight to cap
            scale = sector_cap / sec[s]
            w.loc[idx] = w.loc[idx] * scale
        total = w.sum()
        if total > 0:
            w = w / total
    return w


def hhi(w: pd.Series) -> float:
    return float((w**2).sum())


def main():
    pq = REPORTS / "positions_daily.parquet"
    if not pq.exists():
        raise SystemExit(
            "Missing reports\\positions_daily.parquet — run bootstrap first."
        )
    pos = pd.read_parquet(pq)
    pos["date"] = pd.to_datetime(pos["date"])
    last = pos.loc[pos["date"].idxmax()]
    d = last["date"]
    snap = pos[pos["date"] == d].copy()

    w0 = snap.set_index("ticker")["weight"].clip(lower=0.0)
    w0 = w0 / w0.sum() if w0.sum() > 0 else w0
    sector = w0.index.to_series().map(SECTOR_MAP).fillna("Other")

    pol = load_yaml(CONFIG) if CONFIG.exists() else {}
    sec_cap = float(pol.get("sector_cap_base_pct", 35)) / 100.0
    name_cap = 0.12  # 12% per-name cap

    w_capped = cap_by_name(w0.copy(), name_cap)
    w_final = enforce_sector_caps(w_capped.copy(), sector, sec_cap, iters=50)

    comp = pd.DataFrame(
        {
            "ticker": w0.index,
            "sector": sector.values,
            "w_current": w0.values,
            "w_capped": w_capped.reindex(w0.index).fillna(0.0).values,
            "w_final": w_final.reindex(w0.index).fillna(0.0).values,
        }
    ).sort_values("w_final", ascending=False)

    comp["delta_vs_current"] = comp["w_final"] - comp["w_current"]
    comp.to_csv(OUT, index=False)

    # portfolio metrics
    df = comp.copy()
    sec_current = df.groupby("sector")["w_current"].sum().to_dict()
    sec_final = df.groupby("sector")["w_final"].sum().to_dict()
    metrics = {
        "date": str(d.date()),
        "hhi_current": round(hhi(df["w_current"]), 6),
        "hhi_final": round(hhi(df["w_final"]), 6),
        "max_name_current": round(float(df["w_current"].max()), 4),
        "max_name_final": round(float(df["w_final"].max()), 4),
        "sector_cap": sec_cap,
        "per_name_cap": name_cap,
        "sector_current": {k: round(float(v), 4) for k, v in sec_current.items()},
        "sector_final": {k: round(float(v), 4) for k, v in sec_final.items()},
    }
    print(json.dumps({"out_csv": str(OUT), "metrics": metrics}, indent=2))

    open_win(OUT)
    open_win(CONFIG)


if __name__ == "__main__":
    main()

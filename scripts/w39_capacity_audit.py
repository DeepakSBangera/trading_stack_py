from __future__ import annotations

import json
import math
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import numpy as np

# --- repo roots
ROOT = Path(r"F:\Projects\trading_stack_py")
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
DOCS = ROOT / "docs"

# Inputs (we try these in order; first one found is used)
TARGETS_W11 = REPORTS / "wk11_blend_targets.csv"   # columns: date,ticker,target_w,...
ORDERS_W12  = REPORTS / "wk12_orders_lastday.csv"  # columns: ticker,side,qty,px_ref,notional_ref,...
WIRES_W19   = REPORTS / "wires" / "w19_orders_wire.csv"
SECTOR_MAP  = REPORTS / "sectors_map.csv"          # optional: ticker,sector
# Price panel folder (optional, for ADV from OHLC if present)
PRICES_DIR  = ROOT / "data" / "prices"

OUT_CSV     = REPORTS / "wk39_capacity_audit.csv"
OUT_SUMMARY = REPORTS / "wk39_capacity_summary.json"

# Defaults
DEFAULT_NOTIONAL_INR = 10_000_000.0  # portfolio notional fallback
FALLBACK_ADV_INR     = 5_000_000.0   # if we can't infer ADV per ticker
MAX_PCT_ADV_CAP      = 0.10          # recommended per-order ADV cap (10%) baseline
SECTOR_CAP           = 0.35          # sector NAV cap (35%)
NAME_CAP             = 0.06          # per-name NAV cap (6%)

def _pick_first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None

def _load_targets_or_orders() -> pd.DataFrame:
    """
    Returns a DataFrame of latest intended positions or orders with:
    ticker, target_w (weight), notional (optional), px_ref (optional)
    """
    # 1) wk11 targets
    if TARGETS_W11.exists():
        df = pd.read_csv(TARGETS_W11, parse_dates=["date"])
        last = df["date"].max()
        out = df[df["date"] == last].copy()
        out = out.rename(columns={"target_w": "weight"})
        out["weight"] = pd.to_numeric(out["weight"], errors="coerce").fillna(0.0)
        out = out[["ticker", "weight"]].drop_duplicates()
        return out

    # 2) wk12 orders_lastday
    if ORDERS_W12.exists():
        df = pd.read_csv(ORDERS_W12)
        cols = {c.lower(): c for c in df.columns}
        t = cols.get("ticker", "ticker")
        n = cols.get("notional_ref") or cols.get("notional") or None
        p = cols.get("px_ref") or cols.get("price") or None
        q = cols.get("qty") or None
        out = df[[t]].copy()
        out.columns = ["ticker"]
        out["weight"] = np.nan  # unknown; we’ll evaluate capacity on notional later if available
        if n and n in df.columns:
            out["notional_ref"] = pd.to_numeric(df[n], errors="coerce")
        if p and p in df.columns:
            out["px_ref"] = pd.to_numeric(df[p], errors="coerce")
        if q and q in df.columns:
            out["qty"] = pd.to_numeric(df[q], errors="coerce")
        return out.drop_duplicates(subset=["ticker"])

    # 3) wires W19
    if WIRES_W19.exists():
        df = pd.read_csv(WIRES_W19)
        cols = {c.lower(): c for c in df.columns}
        t = cols.get("ticker", "ticker")
        n = cols.get("notional") or None
        out = df[[t]].copy()
        out.columns = ["ticker"]
        out["weight"] = np.nan
        if n and n in df.columns:
            out["notional_ref"] = pd.to_numeric(df[n], errors="coerce")
        return out.drop_duplicates(subset=["ticker"])

    # 4) empty fallback
    return pd.DataFrame(columns=["ticker", "weight"])

def _adv_from_prices(ticker: str) -> float | None:
    """
    Try to estimate INR ADV using recent 60 trading days from parquet: price*volume.
    File naming assumption: data/prices/{TICKER}.parquet with columns 'date','close','volume'
    """
    p = PRICES_DIR / f"{ticker}.parquet"
    if not p.exists():
        return None
    try:
        x = pd.read_parquet(p)
        cols = {c.lower(): c for c in x.columns}
        c = cols.get("close") or cols.get("px_close") or None
        v = cols.get("volume") or None
        if not c or not v:
            return None
        s = x[[c, v]].dropna().copy()
        s["turnover_inr"] = pd.to_numeric(s[c], errors="coerce") * pd.to_numeric(s[v], errors="coerce")
        s = s.dropna()
        if s.empty:
            return None
        # 60d mean as ADV proxy
        return float(s["turnover_inr"].tail(60).mean())
    except Exception:
        return None

def _load_sector_map() -> dict[str, str]:
    if not SECTOR_MAP.exists():
        return {}
    try:
        df = pd.read_csv(SECTOR_MAP)
        cols = {c.lower(): c for c in df.columns}
        t = cols.get("ticker", "ticker")
        s = cols.get("sector", "sector")
        return {str(r[t]).strip(): str(r[s]).strip() for _, r in df.iterrows() if pd.notna(r.get(t)) and pd.notna(r.get(s))}
    except Exception:
        return {}

def _recommend_caps(audit: pd.DataFrame) -> dict:
    """
    Recommend per-name ADV% cap and sector cap based on liquidity distribution.
    """
    if audit.empty:
        return {
            "name_cap_nav": NAME_CAP,
            "sector_cap_nav": SECTOR_CAP,
            "per_order_pct_adv_cap": MAX_PCT_ADV_CAP,
        }
    advs = audit["adv_inr"].replace([np.inf, -np.inf], np.nan).dropna()
    if advs.empty:
        return {
            "name_cap_nav": NAME_CAP,
            "sector_cap_nav": SECTOR_CAP,
            "per_order_pct_adv_cap": MAX_PCT_ADV_CAP,
        }
    # If median ADV is small, tighten per-order cap; else keep baseline.
    med_adv = float(advs.median())
    pct_cap = 0.05 if med_adv < 3_000_000 else (0.10 if med_adv < 15_000_000 else 0.15)
    return {
        "name_cap_nav": NAME_CAP,
        "sector_cap_nav": SECTOR_CAP,
        "per_order_pct_adv_cap": pct_cap,
    }

def main():
    REPORTS.mkdir(parents=True, exist_ok=True)

    # Source positions/orders
    df = _load_targets_or_orders()
    if df.empty:
        info = {
            "error": "No inputs found (wk11 targets or wk12/w19 orders).",
            "searched": [str(TARGETS_W11), str(ORDERS_W12), str(WIRES_W19)],
        }
        OUT_SUMMARY.write_text(json.dumps(info, indent=2), encoding="utf-8")
        print(json.dumps(info, indent=2))
        return

    # Estimate portfolio notional if weights present; else sum order notionals if available.
    have_weights = "weight" in df.columns and df["weight"].notna().any()
    if have_weights:
        portfolio_notional = DEFAULT_NOTIONAL_INR
    else:
        n = df.get("notional_ref")
        portfolio_notional = float(n.sum()) if n is not None and pd.notna(n).any() else DEFAULT_NOTIONAL_INR

    # ADV inference
    adv_list = []
    for t in sorted(set(df["ticker"].astype(str))):
        adv = _adv_from_prices(t)
        adv_list.append((t, FALLBACK_ADV_INR if adv is None or math.isnan(adv) or adv <= 0 else adv))
    adv_df = pd.DataFrame(adv_list, columns=["ticker", "adv_inr"])

    # Join
    merged = df.merge(adv_df, on="ticker", how="left")
    # If we have weights, infer per-name notional; else use order notional if any
    if have_weights:
        merged["w_eff"] = pd.to_numeric(merged["weight"], errors="coerce").fillna(0.0).clip(lower=0.0)
        merged["notional_inr"] = merged["w_eff"] * portfolio_notional
    else:
        merged["notional_inr"] = pd.to_numeric(merged.get("notional_ref"), errors="coerce")

    # Percent of ADV for the implied trade size (if qty/px given, notional_ref covers it)
    merged["pct_adv_for_trade"] = (merged["notional_inr"] / merged["adv_inr"]).replace([np.inf, -np.inf], np.nan)
    merged["pct_adv_for_trade"] = merged["pct_adv_for_trade"].fillna(0.0)

    # Simple market-impact proxy: slippage_bps ≈ k * sqrt(pct_of_adv)
    # (Kissell-style roughness; purely indicative)
    K = 35.0
    merged["slippage_bps_est"] = (K * np.sqrt(merged["pct_adv_for_trade"].clip(lower=0.0))).round(3)

    # Stress curve: 1% → 20% of ADV
    stress = []
    for _, r in merged.iterrows():
        t = str(r["ticker"])
        adv = float(r["adv_inr"])
        for p in [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]:
            slip = K * np.sqrt(p)
            trade_inr = p * adv
            stress.append((t, p, trade_inr, round(slip, 3)))
    stress_df = pd.DataFrame(stress, columns=["ticker", "pct_of_adv", "trade_inr", "slippage_bps_est"])

    # Sector grouping if available
    sector_map = _load_sector_map()
    merged["sector"] = merged["ticker"].map(sector_map).fillna("UNKNOWN")

    # Save audit
    merged = merged[["ticker", "sector", "adv_inr", "notional_inr", "pct_adv_for_trade", "slippage_bps_est"]]
    merged.to_csv(OUT_CSV, index=False)

    rec = _recommend_caps(merged)
    summary = {
        "as_of_ist": datetime.now().astimezone().isoformat(),
        "rows": int(merged.shape[0]),
        "portfolio_notional_inr": portfolio_notional,
        "recommendations": rec,
        "files": {
            "audit_csv": str(OUT_CSV),
        },
        "notes": "ADV estimated from prices when available; otherwise fallback used. Slippage is indicative.",
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    # Also write stress table alongside the audit for manual inspection
    stress_path = REPORTS / "wk39_capacity_stress.csv"
    stress_df.to_csv(stress_path, index=False)

if __name__ == "__main__":
    main()

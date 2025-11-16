# scripts/w12_emit_canonical_orders.py  (v2)
from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path(r"F:\Projects\trading_stack_py")
REPORTS = ROOT / "reports"
DATA = ROOT / "data" / "prices"

TARGETS_CSV = REPORTS / "wk11_blend_targets.csv"  # input from W11
OUT_CANON = REPORTS / "wk12_orders_lastday_canonical.csv"  # standard schema for W13
OUT_LASTDAY = REPORTS / "wk12_orders_lastday.csv"  # overwrite to same schema
WHY_CSV = REPORTS / "wk12_emit_canonical_orders_why_dropped.csv"
DIAG_JSON = REPORTS / "w12_emit_canonical_orders_diag.json"

# --- knobs ---
PORTFOLIO_NOTIONAL_INR = 10_000_000  # ₹1 cr
WEIGHT_ABS_MIN = 1e-6  # ignore truly tiny weights
MIN_QTY = 1  # avoid rounding to 0
PRICE_COL_CANDS = ["close", "px_close", "price"]
DATE_CANDS = ["date", "dt", "trading_day", "asof", "as_of"]
TICKER_CANDS = ["ticker", "symbol", "name", "secid", "instrument"]
WEIGHT_CANDS = [
    "target_w",
    "weight",
    "w",
    "blend_w",
    "final_w",
    "target_weight",
    "weight_target",
    "w_target",
]


def _pick(df: pd.DataFrame, cands: list[str]) -> str | None:
    low = {c.lower(): c for c in df.columns}
    for k in cands:
        if k in low:
            return low[k]
    for c in df.columns:
        lc = c.lower().replace(" ", "").replace("-", "_")
        for k in cands:
            if lc == k.replace(" ", "").replace("-", "_"):
                return c
    return None


def _lookup_close(ticker: str, d: dt.date) -> float | None:
    p = DATA / f"{ticker}.parquet"
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p)
        # pick columns flexibly
        cols = {c.lower(): c for c in df.columns}
        dcol = cols.get("date") or cols.get("dt")
        ccol = None
        for k in PRICE_COL_CANDS:
            if k in cols:
                ccol = cols[k]
                break
        if not dcol or not ccol:
            return None
        df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
        df = df.sort_values(dcol)
        tgt = pd.to_datetime(d)
        exact = df[df[dcol] == tgt]
        if not exact.empty:
            v = float(pd.to_numeric(exact.iloc[0][ccol], errors="coerce"))
            return v if math.isfinite(v) and v > 0 else None
        prev = df[df[dcol] < tgt]
        if not prev.empty:
            v = float(pd.to_numeric(prev.iloc[-1][ccol], errors="coerce"))
            return v if math.isfinite(v) and v > 0 else None
        return None
    except Exception:
        return None


def main():
    diag = {
        "source": str(TARGETS_CSV),
        "mapping": {},
        "rows_raw": 0,
        "rows_out": 0,
        "last_day": None,
    }
    why = []

    if not TARGETS_CSV.exists():
        raise SystemExit(f"Missing {TARGETS_CSV}. Run W11 first.")

    df = pd.read_csv(TARGETS_CSV)
    if df.empty:
        raise SystemExit("wk11_blend_targets.csv is empty.")

    dcol = _pick(df, DATE_CANDS)
    tcol = _pick(df, TICKER_CANDS)
    wcol = _pick(df, WEIGHT_CANDS)
    if not dcol or not tcol or not wcol:
        raise SystemExit(
            f"Missing required columns. date:{dcol} ticker:{tcol} weight:{wcol}"
        )

    diag["mapping"] = {"date": dcol, "ticker": tcol, "weight": wcol}

    df[dcol] = pd.to_datetime(df[dcol], errors="coerce").dt.date
    df[wcol] = pd.to_numeric(df[wcol], errors="coerce")
    diag["rows_raw"] = int(df.shape[0])

    last_day = df[dcol].dropna().max()
    if pd.isna(last_day):
        raise SystemExit("Could not determine last trading day from targets file.")
    diag["last_day"] = str(last_day)

    last = df[df[dcol] == last_day].copy()
    rows = []
    for _, r in last.iterrows():
        ticker = str(r[tcol])
        w = float(r[wcol]) if not pd.isna(r[wcol]) else 0.0
        if abs(w) < WEIGHT_ABS_MIN:
            why.append({"ticker": ticker, "why": "tiny_weight", "weight": w})
            continue

        side = "BUY" if w >= 0 else "SELL"
        px = _lookup_close(ticker, last_day)
        if px is None or px <= 0:
            why.append({"ticker": ticker, "why": "no_price", "weight": w})
            continue

        notional = abs(w) * PORTFOLIO_NOTIONAL_INR
        qty = int(round(notional / px))
        if qty < MIN_QTY:
            qty = MIN_QTY

        rows.append(
            {
                "date": last_day,
                "ticker": ticker,
                "side": side,
                "qty": qty,
                "px_ref": float(px),
            }
        )

    out = pd.DataFrame(rows, columns=["date", "ticker", "side", "qty", "px_ref"])
    REPORTS.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CANON, index=False)
    out.to_csv(OUT_LASTDAY, index=False)
    if why:
        pd.DataFrame(why).to_csv(WHY_CSV, index=False)

    diag["rows_out"] = int(out.shape[0])
    with DIAG_JSON.open("w", encoding="utf-8") as f:
        json.dump(diag, f, indent=2)

    print(
        json.dumps(
            {
                "last_trading_day": str(last_day),
                "rows": int(out.shape[0]),
                "out_canonical": str(OUT_CANON),
                "out_lastday": str(OUT_LASTDAY),
                "why_csv": str(WHY_CSV) if why else None,
                "diag_json": str(DIAG_JSON),
                "notional_inr": PORTFOLIO_NOTIONAL_INR,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

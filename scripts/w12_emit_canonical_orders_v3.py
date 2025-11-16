# scripts/w12_emit_canonical_orders_v3.py
from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path(r"F:\Projects\trading_stack_py")
REPORTS = ROOT / "reports"
DATA = ROOT / "data" / "prices"

TARGETS_CSV = REPORTS / "wk11_blend_targets.csv"
OUT_CANON = REPORTS / "wk12_orders_lastday_canonical.csv"
OUT_LASTDAY = REPORTS / "wk12_orders_lastday.csv"
WHY_CSV = REPORTS / "wk12_emit_canonical_orders_why_dropped.csv"
DIAG_JSON = REPORTS / "w12_emit_canonical_orders_diag.json"

# ----- knobs -----
PORTFOLIO_NOTIONAL_INR = 10_000_000
WEIGHT_ABS_MIN = 1e-6
MIN_QTY = 1
ALLOW_SYNTHETIC_PX = True  # set to False when real prices are wired
SYNTHETIC_PX_DEFAULT = 100.0

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
    "sum_target_w",
    "sum_w",
    "wt",
    "weight_final",
]
PRICE_COL_CANDS = ["close", "px_close", "price"]


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


def _auto_weight_col(df: pd.DataFrame) -> str | None:
    # if none of the standard names matched, pick any float column with values in [-1,1] and non-zero variance
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() == 0:
            continue
        if s.abs().max() <= 1.0001 and s.abs().sum() > 0:
            return c
    return None


def _lookup_close(ticker: str, d: dt.date) -> float | None:
    p = DATA / f"{ticker}.parquet"
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p)
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
        "px_source": "prices|synthetic" if ALLOW_SYNTHETIC_PX else "prices",
    }
    why = []

    if not TARGETS_CSV.exists():
        raise SystemExit(f"Missing {TARGETS_CSV}. Run W11 first.")

    df = pd.read_csv(TARGETS_CSV)
    if df.empty:
        raise SystemExit("wk11_blend_targets.csv is empty.")

    dcol = _pick(df.columns, DATE_CANDS)
    tcol = _pick(df.columns, TICKER_CANDS)
    wcol = _pick(df.columns, WEIGHT_CANDS) or _auto_weight_col(df)
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
        px_src = "prices"
        if (px is None or px <= 0) and ALLOW_SYNTHETIC_PX:
            px = SYNTHETIC_PX_DEFAULT
            px_src = "synthetic"

        if px is None or px <= 0:
            why.append({"ticker": ticker, "why": "no_price"})
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
                "px_source": px_src,
            }
        )

    out = pd.DataFrame(
        rows, columns=["date", "ticker", "side", "qty", "px_ref", "px_source"]
    )
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
                "px_fallback": ALLOW_SYNTHETIC_PX,
                "why_csv": str(WHY_CSV) if why else None,
                "diag_json": str(DIAG_JSON),
                "notional_inr": PORTFOLIO_NOTIONAL_INR,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

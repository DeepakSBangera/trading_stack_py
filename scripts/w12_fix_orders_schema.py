# scripts/w12_fix_orders_schema.py
from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"F:\Projects\trading_stack_py")
REPORTS = ROOT / "reports"
DATA = ROOT / "data" / "prices"

IN_CSV = REPORTS / "wk12_orders_lastday.csv"  # source from W12
SCHED_CSV = REPORTS / "wk12_orders_schedule.csv"  # optional helper to infer date
OUT_CSV = REPORTS / "wk12_orders_lastday_canonical.csv"  # standardized for W13
DIAG_JSON = REPORTS / "wk12_orders_lastday_canonical_diag.json"
WHY_CSV = REPORTS / "wk12_orders_lastday_canonical_why_dropped.csv"

DEFAULT_PORTFOLIO_NOTIONAL_INR = 10_000_000  # ₹1 cr

# broader candidates
PRICE_CANDS = [
    "px_ref",
    "ref_price",
    "price",
    "px_close",
    "close",
    "last",
    "close_price",
    "ref",
    "reference_price",
    "mid",
    "vwap",
    "open",
    "px",
]
QTY_CANDS = [
    "qty",
    "quantity",
    "shares",
    "shares_delta",
    "delta_qty",
    "qty_intent",
    "order_qty",
    "trade_qty",
    "exec_qty",
]
SIDE_CANDS = ["side", "action", "buy_sell", "order_side", "direction", "side_flag"]
NOTIONAL_CANDS = [
    "notional",
    "notional_delta",
    "order_notional",
    "value",
    "notional_ref",
    "notional_target",
    "dollar_value",
    "cash_value",
]
WEIGHT_CANDS = [
    "weight",
    "w",
    "target_w",
    "target_weight",
    "weight_target",
    "w_target",
    "delta_w",
    "w_change",
    "final_weight",
    "w_final",
]

TICKER_CANDS = ["ticker", "symbol", "name", "secid", "instrument", "security", "ric"]
DATE_CANDS = ["date", "trading_day", "dt", "asof", "as_of", "trade_date"]


def _pick(df: pd.DataFrame, cands):
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
        cols = {c.lower(): c for c in df.columns}
        dcol = cols.get("date") or cols.get("dt")
        ccol = cols.get("close") or cols.get("px_close") or cols.get("price")
        if not dcol or not ccol:
            return None
        df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
        df = df.sort_values(dcol)
        tgt = pd.to_datetime(d)
        m = df[df[dcol] == tgt]
        if not m.empty:
            v = float(pd.to_numeric(m.iloc[0][ccol], errors="coerce"))
            return v if math.isfinite(v) and v > 0 else None
        prev = df[df[dcol] < tgt]
        if not prev.empty:
            v = float(pd.to_numeric(prev.iloc[-1][ccol], errors="coerce"))
            return v if math.isfinite(v) and v > 0 else None
        return None
    except Exception:
        return None


def _norm_side(x):
    if isinstance(x, (int, float)) and not pd.isna(x):
        if float(x) > 0:
            return "BUY"
        if float(x) < 0:
            return "SELL"
    if not isinstance(x, str):
        return None
    v = x.strip().upper()
    if v in ("B", "BUY", "LONG", "BUY_TO_OPEN", "OPEN_BUY"):
        return "BUY"
    if v in ("S", "SELL", "SHORT", "SELL_TO_CLOSE", "CLOSE_SELL"):
        return "SELL"
    return None


def _coerce_num(s, default=None):
    out = pd.to_numeric(s, errors="coerce")
    if default is not None:
        out = out.fillna(default)
    return out


def _infer_session_date() -> dt.date | None:
    """Use max date from wk12_orders_schedule.csv if available."""
    if not SCHED_CSV.exists():
        return None
    try:
        sch = pd.read_csv(SCHED_CSV)
        dcol = _pick(sch, DATE_CANDS)
        if not dcol:
            return None
        d = pd.to_datetime(sch[dcol], errors="coerce").dt.date
        d = d.dropna()
        return d.max() if not d.empty else None
    except Exception:
        return None


def main():
    diag = {
        "source": str(IN_CSV),
        "found_cols": [],
        "mapping": {},
        "row_counts": {},
        "notes": [],
    }

    if not IN_CSV.exists():
        raise SystemExit(f"Missing {IN_CSV}. Run W12 first.")

    src = pd.read_csv(IN_CSV)
    diag["found_cols"] = list(src.columns)
    if src.empty:
        raise SystemExit("W12 file is empty.")

    why_rows = []

    # --- date
    dcol = _pick(src, DATE_CANDS)
    if dcol:
        src["date_norm"] = pd.to_datetime(src[dcol], errors="coerce").dt.date
        diag["mapping"]["date"] = dcol
    else:
        # use schedule max date if no per-row date
        guess = _infer_session_date()
        if guess is not None:
            src["date_norm"] = pd.Series([guess] * len(src), dtype="object")
            diag["mapping"]["date"] = f"[filled] {guess.isoformat()} from wk12_orders_schedule.csv"
            diag["notes"].append("No date column in W12 file; filled using max date from wk12_orders_schedule.csv.")
        else:
            src["date_norm"] = pd.NaT
            diag["mapping"]["date"] = None
            diag["notes"].append("No date column and could not infer date from schedule.")

    # --- ticker
    tcol = _pick(src, TICKER_CANDS)
    if not tcol:
        raise SystemExit("No ticker/symbol column found in W12 file.")
    src["ticker_norm"] = src[tcol].astype(str)
    diag["mapping"]["ticker"] = tcol

    # --- price: direct then parquet fallback
    pcol = _pick(src, PRICE_CANDS)
    price = _coerce_num(src[pcol], np.nan) if pcol else pd.Series([np.nan] * len(src))
    if pcol:
        diag["mapping"]["px_ref"] = pcol

    if price.isna().any():
        filled = 0
        for i in price[price.isna()].index:
            d = src.at[i, "date_norm"]
            t = str(src.at[i, "ticker_norm"])
            if pd.isna(d):
                continue
            v = _lookup_close(t, d)
            if v is not None:
                price.at[i] = v
                filled += 1
        if filled > 0:
            diag["notes"].append(f"Filled {filled} px_ref via data\\prices parquet close fallback.")
    src["px_ref"] = _coerce_num(price, np.nan)

    # --- qty / notional / weight
    qcol = _pick(src, QTY_CANDS)
    ncol = _pick(src, NOTIONAL_CANDS)
    wcol = _pick(src, WEIGHT_CANDS)

    qty_raw = _coerce_num(src[qcol]) if qcol else None
    notional_raw = _coerce_num(src[ncol]) if ncol else None
    weight_raw = _coerce_num(src[wcol]) if wcol else None
    if qcol:
        diag["mapping"]["qty"] = qcol
    if ncol:
        diag["mapping"]["notional"] = ncol
    if wcol:
        diag["mapping"]["weight"] = wcol

    # --- side: direct or infer from sign
    scol = _pick(src, SIDE_CANDS)
    side = src[scol].map(_norm_side) if scol else pd.Series([None] * len(src))
    if scol:
        diag["mapping"]["side"] = scol

    def infer_side(i):
        if isinstance(side.iat[i], str):
            return side.iat[i]
        for s in (qty_raw, notional_raw, weight_raw):
            if s is not None and pd.notna(s.iat[i]) and float(s.iat[i]) != 0:
                return "BUY" if float(s.iat[i]) > 0 else "SELL"
        return "BUY"

    side_final = [infer_side(i) for i in range(len(src))]

    # --- qty_abs
    if qty_raw is not None:
        qty_abs = qty_raw.abs().round().astype("Int64")
        qty_source = "qty"
    elif notional_raw is not None:
        qty_abs = (notional_raw.abs() / src["px_ref"].replace(0, np.nan)).round().astype("Int64")
        qty_source = "notional/px_ref"
    elif weight_raw is not None:
        notional = weight_raw.fillna(0.0) * DEFAULT_PORTFOLIO_NOTIONAL_INR
        qty_abs = (notional.abs() / src["px_ref"].replace(0, np.nan)).round().astype("Int64")
        qty_source = "weight×portfolio_notional/px_ref"
    else:
        qty_abs = pd.Series([pd.NA] * len(src), dtype="Int64")
        qty_source = "none"
    diag["mapping"]["qty_abs_source"] = qty_source

    # --- assemble + collect drop reasons
    out = pd.DataFrame(
        {
            "date": src["date_norm"],
            "ticker": src["ticker_norm"],
            "side": side_final,
            "qty": qty_abs.astype("Int64"),
            "px_ref": src["px_ref"],
        }
    )

    def reason_row(i):
        reasons = []
        r = out.iloc[i]
        if pd.isna(r["date"]):
            reasons.append("no_date")
        if pd.isna(r["ticker"]) or str(r["ticker"]).strip() == "":
            reasons.append("no_ticker")
        if pd.isna(r["side"]):
            reasons.append("no_side")
        if pd.isna(r["qty"]) or (pd.notna(r["qty"]) and int(r["qty"]) <= 0):
            reasons.append("bad_qty")
        if pd.isna(r["px_ref"]) or (pd.notna(r["px_ref"]) and float(r["px_ref"]) <= 0):
            reasons.append("bad_px")
        return ";".join(reasons) if reasons else ""

    for i in range(len(out)):
        rr = reason_row(i)
        if rr:
            why_rows.append(
                {
                    "i": i,
                    "ticker": out.iloc[i]["ticker"],
                    "date": str(out.iloc[i]["date"]),
                    "side": out.iloc[i]["side"],
                    "qty": out.iloc[i]["qty"],
                    "px_ref": out.iloc[i]["px_ref"],
                    "why_dropped": rr,
                }
            )

    diag["row_counts"]["raw"] = int(out.shape[0])
    out = out.dropna(subset=["date", "ticker", "side", "qty", "px_ref"])
    diag["row_counts"]["after_dropna"] = int(out.shape[0])
    out = out[(out["qty"] > 0) & (out["px_ref"] > 0)]
    diag["row_counts"]["final"] = int(out.shape[0])

    REPORTS.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    if why_rows:
        pd.DataFrame(why_rows).to_csv(WHY_CSV, index=False)
    with DIAG_JSON.open("w", encoding="utf-8") as f:
        json.dump(diag, f, indent=2, default=str)

    print(
        json.dumps(
            {
                "canonical_out": str(OUT_CSV),
                "rows": int(out.shape[0]),
                "diag": str(DIAG_JSON),
                "why_dropped_csv": str(WHY_CSV) if why_rows else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

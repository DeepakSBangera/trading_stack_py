# scripts/w19_wire_orders.py
from __future__ import annotations

import datetime as dt
import json
import math
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"F:\Projects\trading_stack_py")
REPORTS = ROOT / "reports"
WIRES = REPORTS / "wires"
WIRES.mkdir(parents=True, exist_ok=True)

# inputs
OK_ORDERS_CSV = REPORTS / "w18_pretrade_ok.csv"  # from W18
FILLS_CSV_OPT = REPORTS / "wk13_dryrun_fills.csv"  # optional — to carry px_fill
PRICES_DIR = ROOT / "data" / "prices"  # optional px fallback

# outputs
WIRE_CSV = WIRES / "w19_orders_wire.csv"
SUMMARY_JSON = WIRES / "w19_wire_summary.json"

# defaults
DEFAULT_TIF = "DAY"
DEFAULT_ORDTYPE = "LIMIT"  # could switch to "MARKET" later
SLIPPAGE_PAD_BPS = 5.0  # add to px_ref when BUY, subtract when SELL for limit buffer


def _bps(x: float) -> float:
    return x / 10000.0


def _safe_num(x, default=np.nan) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _lookup_last_close(ticker: str, date_val: dt.date) -> float | None:
    p = PRICES_DIR / f"{ticker}.parquet"
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
        t = pd.Timestamp(date_val)
        hit = df[df[dcol] == t]
        if not hit.empty:
            val = float(pd.to_numeric(hit.iloc[0][ccol], errors="coerce"))
            return val if math.isfinite(val) and val > 0 else None
        prev = df[df[dcol] < t]
        if not prev.empty:
            val = float(pd.to_numeric(prev.iloc[-1][ccol], errors="coerce"))
            return val if math.isfinite(val) and val > 0 else None
        return None
    except Exception:
        return None


def _load_ok_orders() -> pd.DataFrame:
    if not OK_ORDERS_CSV.exists():
        raise FileNotFoundError(f"Missing {OK_ORDERS_CSV}. Run W18 first.")
    df = pd.read_csv(OK_ORDERS_CSV)
    # expected cols: date, ticker, side, qty, px_ref
    cols = {c.lower(): c for c in df.columns}
    need = ["date", "ticker", "side", "qty", "px_ref"]
    missing = [n for n in need if n not in cols]
    if missing:
        raise ValueError(f"{OK_ORDERS_CSV} missing columns: {missing}")
    df = df.rename(
        columns={
            cols["date"]: "date",
            cols["ticker"]: "ticker",
            cols["side"]: "side",
            cols["qty"]: "qty",
            cols["px_ref"]: "px_ref",
        }
    )
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(0).astype(int).abs()
    df["side"] = df["side"].astype(str).str.upper().str.strip()
    df["px_ref"] = pd.to_numeric(df["px_ref"], errors="coerce")
    return df


def _merge_fills(df: pd.DataFrame) -> pd.DataFrame:
    if not FILLS_CSV_OPT.exists():
        df["px_fill"] = np.nan
        return df
    f = pd.read_csv(FILLS_CSV_OPT)
    c = {x.lower(): x for x in f.columns}
    for need in ["date", "ticker", "px_fill"]:
        if need not in c:
            df["px_fill"] = np.nan
            return df
    f = f.rename(
        columns={c["date"]: "date", c["ticker"]: "ticker", c["px_fill"]: "px_fill"}
    )
    f["date"] = pd.to_datetime(f["date"], errors="coerce").dt.date
    return df.merge(f[["date", "ticker", "px_fill"]], on=["date", "ticker"], how="left")


def _limit_price(side: str, ref: float, fill: float | None) -> float:
    base = fill if (fill is not None and math.isfinite(fill) and fill > 0) else ref
    if not (math.isfinite(base) and base > 0):
        return np.nan
    pad = _bps(SLIPPAGE_PAD_BPS) * base
    return round(base + pad if side == "BUY" else base - pad, 4)


def main():
    WIRES.mkdir(parents=True, exist_ok=True)
    orders = _load_ok_orders()
    orders = _merge_fills(orders)

    # fallback price if both ref & fill are NaN – use last close
    fallback_idx = orders[
        ~orders["px_ref"].apply(math.isfinite)
        & ~orders["px_fill"].apply(lambda x: math.isfinite(x) if x == x else False)
    ].index
    for i in fallback_idx:
        px = _lookup_last_close(orders.at[i, "ticker"], orders.at[i, "date"])
        if px is not None:
            orders.at[i, "px_ref"] = px

    rows = []
    for _, r in orders.iterrows():
        side = r["side"]
        qty = int(abs(r["qty"]))
        ref = _safe_num(r["px_ref"])
        fill = _safe_num(r.get("px_fill", np.nan))
        lmt = _limit_price(side, ref, fill)

        coid = f"DRY-{uuid.uuid4().hex[:12].upper()}"
        rows.append(
            {
                "client_order_id": coid,
                "date": r["date"].isoformat(),
                "symbol": r["ticker"],
                "side": side,  # BUY / SELL
                "quantity": qty,
                "order_type": DEFAULT_ORDTYPE,  # LIMIT
                "limit_price": lmt,
                "time_in_force": DEFAULT_TIF,  # DAY
                "ref_price": ref,
                "px_fill_hint": fill if math.isfinite(fill) else "",
                "notes": "dry_run",
            }
        )

    wire = pd.DataFrame(rows)
    wire = wire.sort_values(["date", "symbol", "side"]).reset_index(drop=True)
    wire.to_csv(WIRE_CSV, index=False)

    summary = {
        "rows": int(wire.shape[0]),
        "wire_csv": str(WIRE_CSV),
        "limit_price_pad_bps": SLIPPAGE_PAD_BPS,
        "order_type": DEFAULT_ORDTYPE,
        "tif": DEFAULT_TIF,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

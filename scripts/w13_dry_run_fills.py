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

# --- knobs ---
DEFAULT_COMMISSION_BPS = 1.5
DEFAULT_TAX_BPS = 0.0
MIN_SLIPPAGE_BPS = 2.0
BASE_IMPACT_BPS = 8.0
IMPACT_EXPONENT = 0.65
PARTICIPATION_CAP_PCT = 12.5
VWAP_DRIFT_BPS = 3.0

# If file has weights but not qty/notional, we’ll use this portfolio notional:
DEFAULT_PORTFOLIO_NOTIONAL_INR = 10_000_000  # ₹1 cr; adjust to your test size

ORDERS_CSV = REPORTS / "wk12_orders_lastday.csv"
ADV_CSV_OPT = REPORTS / "adv_value.parquet"  # optional parquet with per-ticker ADV INR
FILLS_CSV = REPORTS / "wk13_dryrun_fills.csv"
TCA_CSV = REPORTS / "wk13_tca_summary.csv"

# ----- helpers -----
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
]
SIDE_CANDS = ["side", "action", "buy_sell", "order_side", "direction"]
NOTIONAL_CANDS = [
    "notional",
    "notional_delta",
    "order_notional",
    "value",
    "notional_ref",
    "notional_target",
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
]


def _pick_col(df: pd.DataFrame, cands: list[str]) -> str | None:
    cols = {c.lower(): c for c in df.columns}
    for k in cands:
        if k in cols:
            return cols[k]
    # loose match
    for c in df.columns:
        lc = c.lower().replace(" ", "").replace("-", "_")
        for k in cands:
            kk = k.replace(" ", "").replace("-", "_")
            if lc == kk:
                return c
    return None


def _load_adv_map() -> dict[str, float]:
    try:
        p = ADV_CSV_OPT
        if p.exists():
            adv = pd.read_parquet(p)
            cols = {c.lower(): c for c in adv.columns}
            tcol = cols.get("ticker")
            acol = cols.get("adv_value")
            if tcol and acol:
                return dict(
                    zip(
                        adv[tcol],
                        pd.to_numeric(adv[acol], errors="coerce").fillna(0.0),
                        strict=False,
                    )
                )
    except Exception:
        pass
    return {}


def _bps(x: float) -> float:
    return x / 10000.0


def _normalize_side(val: str) -> str | None:
    if not isinstance(val, str):
        return None
    v = val.strip().upper()
    if v in ("B", "BUY", "LONG", "BUY_TO_OPEN", "BUY TO OPEN", "OPEN_BUY"):
        return "BUY"
    if v in ("S", "SELL", "SHORT", "SELL_TO_CLOSE", "SELL TO CLOSE", "CLOSE_SELL"):
        return "SELL"
    return None


def _coerce_numeric(s, default=None):
    out = pd.to_numeric(s, errors="coerce")
    if default is not None:
        out = out.fillna(default)
    return out


def _derive_price_from_notional_qty(df: pd.DataFrame, notional_col: str, qty_series: pd.Series) -> pd.Series:
    notional = _coerce_numeric(df[notional_col], 0.0).astype(float)
    qty_abs = qty_series.abs().replace(0, np.nan)
    px = (notional.abs() / qty_abs).replace([np.inf, -np.inf], np.nan)
    return px


def _lookup_price_from_prices_dir(ticker: str, date_val: dt.date) -> float | None:
    p = DATA / f"{ticker}.parquet"
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p)
        cols = {c.lower(): c for c in df.columns}
        dcol = cols.get("date") or cols.get("dt") or None
        ccol = cols.get("close") or cols.get("px_close") or cols.get("price") or None
        if dcol is None or ccol is None:
            return None
        df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
        df = df.sort_values(dcol)
        tgt = pd.to_datetime(date_val)
        m = df[df[dcol] == tgt]
        if not m.empty:
            val = float(pd.to_numeric(m.iloc[0][ccol], errors="coerce"))
            return val if math.isfinite(val) and val > 0 else None
        prev = df[df[dcol] < tgt]
        if not prev.empty:
            val = float(pd.to_numeric(prev.iloc[-1][ccol], errors="coerce"))
            return val if math.isfinite(val) and val > 0 else None
        return None
    except Exception:
        return None


# ----- core loaders -----
def _load_orders_flex() -> tuple[pd.DataFrame, dict]:
    if not ORDERS_CSV.exists():
        raise FileNotFoundError(f"Missing {ORDERS_CSV}; run W12 first.")

    df = pd.read_csv(ORDERS_CSV)
    map_used: dict[str, str] = {}

    # date
    date_col = _pick_col(df, ["date", "trading_day", "dt"])
    if date_col is None:
        raise ValueError("Could not find a date column (expected one of: date, trading_day, dt)")
    df["date_norm"] = pd.to_datetime(df[date_col], errors="coerce").dt.date
    map_used["date"] = date_col

    # ticker
    tic_col = _pick_col(df, ["ticker", "symbol", "name"])
    if tic_col is None:
        raise ValueError("Could not find a ticker column (expected one of: ticker, symbol, name)")
    df["ticker_norm"] = df[tic_col].astype(str)
    map_used["ticker"] = tic_col

    # side (maybe missing → infer from sign later)
    side_col = _pick_col(df, SIDE_CANDS)
    if side_col is not None:
        side = df[side_col].apply(_normalize_side)
        map_used["side"] = side_col
    else:
        side = pd.Series([None] * len(df))
        map_used["side"] = "inferred from qty/notional/weight sign"

    # price: try columns, else derive (later) or parquet fallback
    price_col = _pick_col(df, PRICE_CANDS)
    price_series = _coerce_numeric(df[price_col], np.nan) if price_col else None
    if price_col:
        map_used["price"] = price_col

    # qty / notional / weight
    qty_col = _pick_col(df, QTY_CANDS)
    notional_col = _pick_col(df, NOTIONAL_CANDS)
    weight_col = _pick_col(df, WEIGHT_CANDS)

    # Priority: qty → notional → weight
    qty_raw = None
    notional_raw = None
    weight_raw = None

    if qty_col is not None:
        qty_raw = _coerce_numeric(df[qty_col], 0.0).astype(float)
        map_used["qty"] = qty_col
    if notional_col is not None:
        notional_raw = _coerce_numeric(df[notional_col], 0.0).astype(float)
        map_used["notional"] = notional_col
    if weight_col is not None:
        weight_raw = _coerce_numeric(df[weight_col], 0.0).astype(float)
        map_used["weight"] = weight_col

    # If none provided, error out with explicit hint
    if qty_raw is None and notional_raw is None and weight_raw is None:
        raise ValueError(
            "No qty/notional/weight columns found.\n"
            f"Qty candidates: {QTY_CANDS}\n"
            f"Notional candidates: {NOTIONAL_CANDS}\n"
            f"Weight candidates: {WEIGHT_CANDS}"
        )

    # If price absent, try derive from (notional/qty)
    if (price_series is None or price_series.isna().all()) and (notional_raw is not None and qty_raw is not None):
        price_series = _derive_price_from_notional_qty(df, notional_col, qty_raw)
        map_used["price"] = f"derived from {notional_col}/{qty_col}"

    # If price still missing, parquet fallback per row
    if price_series is None:
        price_series = pd.Series([np.nan] * len(df), index=df.index, dtype=float)
    if price_series.isna().any():
        for i in price_series[price_series.isna()].index:
            tck = str(df.at[i, "ticker_norm"])
            dte = df.at[i, "date_norm"]
            val = _lookup_price_from_prices_dir(tck, dte)
            if val is not None:
                price_series.at[i] = val
                map_used.setdefault("price_fallback", "data\\prices\\<ticker>.parquet close")

    df["price_norm"] = pd.to_numeric(price_series, errors="coerce")

    # Build qty_abs:
    if qty_raw is not None:
        qty_abs = qty_raw.abs().round().astype(int)
        sign_source = "qty sign"
    elif notional_raw is not None:
        # qty = |notional| / price
        qty_abs = (notional_raw.abs() / df["price_norm"].replace(0, np.nan)).round()
        qty_abs = pd.to_numeric(qty_abs, errors="coerce").fillna(0).astype(int)
        sign_source = "notional sign"
    else:
        # weights only → notional = |weight| × DEFAULT_PORTFOLIO_NOTIONAL_INR
        notional_raw = weight_raw * DEFAULT_PORTFOLIO_NOTIONAL_INR
        qty_abs = (notional_raw.abs() / df["price_norm"].replace(0, np.nan)).round()
        qty_abs = pd.to_numeric(qty_abs, errors="coerce").fillna(0).astype(int)
        sign_source = "weight sign × portfolio notional"

    df["qty_abs"] = qty_abs

    # Infer side if still missing
    def infer_side(row):
        s = row["side_norm"]
        if s in ("BUY", "SELL"):
            return s
        # fallback by precedence: qty → notional → weight
        if qty_raw is not None:
            base = float(qty_raw.loc[row.name])
            if base != 0:
                return "BUY" if base > 0 else "SELL"
        if notional_raw is not None:
            base = float(notional_raw.loc[row.name])
            if base != 0:
                return "BUY" if base > 0 else "SELL"
        if weight_raw is not None:
            base = float(weight_raw.loc[row.name])
            if base != 0:
                return "BUY" if base > 0 else "SELL"
        return "BUY"  # neutral default

    df["side_norm"] = side
    df["side_norm"] = df.apply(infer_side, axis=1)
    map_used["side_inference"] = sign_source

    # sanity filters
    df = df[(df["date_norm"].notna()) & df["ticker_norm"].notna() & df["price_norm"].notna() & (df["price_norm"] > 0)]
    df = df[(df["qty_abs"] > 0)]
    if df.empty:
        raise ValueError("After normalization, no valid orders (check price/qty inference or add weights/notional).")

    return df, map_used


def simulate_fills(orders: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    adv_map = _load_adv_map()
    rows = []

    for _, r in orders.iterrows():
        date = r["date_norm"]
        tick = r["ticker_norm"]
        side = r["side_norm"]
        qty = int(r["qty_abs"])
        px_ref = float(r["price_norm"])

        notional_ref = abs(qty) * px_ref
        adv_val = adv_map.get(tick, np.nan)
        if not math.isfinite(adv_val) or adv_val <= 0:
            adv_val = 5e7  # fallback ADV INR

        part_pct = min(100.0 * notional_ref / adv_val, PARTICIPATION_CAP_PCT)
        rel = max(part_pct / 1.0, 1e-6)
        impact_bps = BASE_IMPACT_BPS * (rel**IMPACT_EXPONENT)
        slippage_bps = MIN_SLIPPAGE_BPS + impact_bps + VWAP_DRIFT_BPS * (1.0 if side == "BUY" else -1.0)

        signed_slip = _bps(slippage_bps) * px_ref * (1 if side == "BUY" else -1)
        px_fill = px_ref + signed_slip

        slip_cost_inr = abs(_bps(slippage_bps) * notional_ref)
        comm_inr = abs(_bps(DEFAULT_COMMISSION_BPS) * notional_ref)
        tax_inr = abs(_bps(DEFAULT_TAX_BPS) * notional_ref)
        tca_inr = slip_cost_inr + comm_inr + tax_inr

        rows.append(
            {
                "date": date,
                "ticker": tick,
                "side": side,
                "qty": qty,
                "px_ref": round(px_ref, 6),
                "px_fill": round(px_fill, 6),
                "slippage_bps": round(slippage_bps, 4),
                "commission_bps": round(DEFAULT_COMMISSION_BPS, 4),
                "tax_bps": round(DEFAULT_TAX_BPS, 4),
                "notional_ref": round(notional_ref, 2),
                "slip_cost_inr": round(slip_cost_inr, 2),
                "comm_inr": round(comm_inr, 2),
                "tax_inr": round(tax_inr, 2),
                "tca_cost_inr": round(tca_inr, 2),
                "participation_pct": round(part_pct, 3),
                "adv_inr": (round(float(adv_val), 2) if math.isfinite(adv_val) else np.nan),
            }
        )

    fills = pd.DataFrame(rows)
    if fills.empty:
        cols = [
            "date",
            "ticker",
            "side",
            "qty",
            "px_ref",
            "px_fill",
            "slippage_bps",
            "commission_bps",
            "tax_bps",
            "notional_ref",
            "slip_cost_inr",
            "comm_inr",
            "tax_inr",
            "tca_cost_inr",
            "participation_pct",
            "adv_inr",
        ]
        fills = pd.DataFrame(columns=cols)

    # summaries
    if not fills.empty:
        grp = fills.groupby("date", as_index=False).agg(
            orders=("ticker", "count"),
            notional=("notional_ref", "sum"),
            tca_cost=("tca_cost_inr", "sum"),
            slip_cost=("slip_cost_inr", "sum"),
            comm=("comm_inr", "sum"),
            tax=("tax_inr", "sum"),
            med_slip_bps=("slippage_bps", "median"),
            p90_slip_bps=("slippage_bps", lambda s: np.percentile(s, 90)),
        )
        overall = {
            "orders": int(fills.shape[0]),
            "notional": float(fills["notional_ref"].sum()),
            "tca_cost": float(fills["tca_cost_inr"].sum()),
            "slip_cost": float(fills["slip_cost_inr"].sum()),
            "comm": float(fills["comm_inr"].sum()),
            "tax": float(fills["tax_inr"].sum()),
            "med_slip_bps": float(fills["slippage_bps"].median()),
            "p90_slip_bps": float(np.percentile(fills["slippage_bps"], 90)),
        }
    else:
        grp = pd.DataFrame(
            columns=[
                "date",
                "orders",
                "notional",
                "tca_cost",
                "slip_cost",
                "comm",
                "tax",
                "med_slip_bps",
                "p90_slip_bps",
            ]
        )
        overall = {
            "orders": 0,
            "notional": 0.0,
            "tca_cost": 0.0,
            "slip_cost": 0.0,
            "comm": 0.0,
            "tax": 0.0,
            "med_slip_bps": 0.0,
            "p90_slip_bps": 0.0,
        }

    # write
    REPORTS.mkdir(parents=True, exist_ok=True)
    fills.to_csv(FILLS_CSV, index=False)
    grp.to_csv(TCA_CSV, index=False)

    summary = {
        "fills_csv": str(FILLS_CSV),
        "tca_csv": str(TCA_CSV),
        "orders": overall["orders"],
        "notional_inr": round(overall["notional"], 2),
        "tca_cost_inr": round(overall["tca_cost"], 2),
        "med_slip_bps": round(overall["med_slip_bps"], 3),
        "p90_slip_bps": round(overall["p90_slip_bps"], 3),
    }
    return fills, grp, summary


def main():
    orders, mapping = _load_orders_flex()
    fills, grp, summary = simulate_fills(orders)
    out = {"mapped_columns": mapping, **summary}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

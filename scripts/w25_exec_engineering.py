from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

# --- paths ---
ROOT = Path(r"F:\Projects\trading_stack_py")
REPORTS = ROOT / "reports"
DATA = ROOT / "data" / "prices"

# Inputs preferred → falls back sensibly
FILLS_IN_OPT = REPORTS / "wk13_dryrun_fills.csv"  # preferred (has px_ref & qty)
ORDERS_LASTDAY = REPORTS / "wk12_orders_lastday.csv"  # fallback
ADV_PARQUET_OPT = REPORTS / "adv_value.parquet"  # optional per-ticker ADV (INR)

# Outputs
DETAIL_CSV = REPORTS / "w25_schedule_compare.csv"
STYLE_SUMMARY_CSV = REPORTS / "w25_tca_by_style.csv"
SUMMARY_CSV = REPORTS / "wk25_exec_engineering.csv"

# --- knobs (aligned with earlier W13 assumptions) ---
DEFAULT_COMMISSION_BPS = 1.5
DEFAULT_TAX_BPS = 0.0
MIN_SLIPPAGE_BPS = 2.0
BASE_IMPACT_BPS = 8.0
IMPACT_EXPONENT = 0.65
VWAP_DRIFT_BPS = 3.0  # VWAP tends to drift with prevailing side
POV_TARGET_PCT = 10.0  # 10% participation-of-volume target
PARTICIPATION_CAP_PCT = 12.5  # hard cap
FALLBACK_ADV_INR = 5e7  # if ADV parquet absent
DEFAULT_PORTFOLIO_NOTIONAL_INR = 10_000_000

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


def _bps(x: float) -> float:
    return x / 10000.0


def _normalize_side(val):
    if isinstance(val, str):
        v = val.strip().upper()
        if v in ("B", "BUY", "LONG", "BUY_TO_OPEN", "BUY TO OPEN", "OPEN_BUY"):
            return "BUY"
        if v in ("S", "SELL", "SHORT", "SELL_TO_CLOSE", "SELL TO CLOSE", "CLOSE_SELL"):
            return "SELL"
    return None


def _pick_col(df: pd.DataFrame, cands: list[str]) -> str | None:
    cols = {c.lower(): c for c in df.columns}
    for k in cands:
        if k in cols:
            return cols[k]
    for c in df.columns:
        lc = c.lower().replace(" ", "").replace("-", "_")
        for k in cands:
            kk = k.replace(" ", "").replace("-", "_")
            if lc == kk:
                return c
    return None


def _lookup_adv_map() -> dict[str, float]:
    try:
        p = ADV_PARQUET_OPT
        if p.exists():
            adv = pd.read_parquet(p)
            tcol = _pick_col(adv, ["ticker", "symbol", "name"])
            vcol = _pick_col(adv, ["adv_value", "adv_inr", "adv"])
            if tcol and vcol:
                return dict(
                    zip(
                        adv[tcol].astype(str),
                        pd.to_numeric(adv[vcol], errors="coerce").fillna(0.0),
                        strict=False,
                    )
                )
    except Exception:
        pass
    return {}


def _lookup_price_from_prices_dir(ticker: str, date_val: dt.date) -> float | None:
    p = DATA / f"{ticker}.parquet"
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p)
        dcol = _pick_col(df, ["date", "dt"])
        ccol = _pick_col(df, ["close", "px_close", "price"])
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
    except Exception:
        return None
    return None


def _load_orders() -> pd.DataFrame:
    # Prefer wk13 fills (already normalized & has px_ref/qty/side)
    if FILLS_IN_OPT.exists():
        df = pd.read_csv(FILLS_IN_OPT)
        # Required columns check
        need = ["date", "ticker", "qty", "px_ref", "side"]
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise ValueError(f"{FILLS_IN_OPT} missing columns: {missing}")
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df["ticker"] = df["ticker"].astype(str)
        df["qty"] = (
            pd.to_numeric(df["qty"], errors="coerce").fillna(0).astype(int).abs()
        )
        df["side"] = df["side"].astype(str)
        df["px_ref"] = pd.to_numeric(df["px_ref"], errors="coerce")
        return df[df["qty"] > 0].copy()

    # Fallback: infer from wk12 lastday
    if not ORDERS_LASTDAY.exists():
        raise FileNotFoundError(f"Missing {FILLS_IN_OPT} and {ORDERS_LASTDAY}")
    raw = pd.read_csv(ORDERS_LASTDAY)
    dcol = _pick_col(raw, ["date", "trading_day", "dt"])
    tcol = _pick_col(raw, ["ticker", "symbol", "name"])
    qcol = _pick_col(raw, QTY_CANDS)
    pcol = _pick_col(raw, PRICE_CANDS)
    scol = _pick_col(raw, SIDE_CANDS)

    if dcol is None or tcol is None:
        raise ValueError("Orders CSV missing date/ticker columns.")
    raw["date"] = pd.to_datetime(raw[dcol], errors="coerce").dt.date
    raw["ticker"] = raw[tcol].astype(str)
    if qcol:
        raw["qty"] = (
            pd.to_numeric(raw[qcol], errors="coerce").fillna(0).abs().astype(int)
        )
    else:
        # Derive qty from weights + default notional + price
        wcol = _pick_col(
            raw,
            ["weight", "w", "target_w", "target_weight", "weight_target", "w_target"],
        )
        if not wcol:
            raise ValueError("No qty and no weight column to infer from.")
        # Price first
        if pcol:
            raw["px_ref"] = pd.to_numeric(raw[pcol], errors="coerce")
        else:
            raw["px_ref"] = np.nan
        for i in raw.index[raw["px_ref"].isna()]:
            pr = _lookup_price_from_prices_dir(
                str(raw.at[i, "ticker"]), raw.at[i, "date"]
            )
            if pr is not None:
                raw.at[i, "px_ref"] = pr
        raw["w"] = pd.to_numeric(raw[wcol], errors="coerce").fillna(0.0)
        notional = (raw["w"] * DEFAULT_PORTFOLIO_NOTIONAL_INR).abs()
        raw["qty"] = (
            (notional / raw["px_ref"].replace(0, np.nan)).round().fillna(0).astype(int)
        )

    if pcol:
        raw["px_ref"] = pd.to_numeric(raw[pcol], errors="coerce")
    else:
        raw["px_ref"] = np.nan
        for i in raw.index[raw["px_ref"].isna()]:
            pr = _lookup_price_from_prices_dir(
                str(raw.at[i, "ticker"]), raw.at[i, "date"]
            )
            if pr is not None:
                raw.at[i, "px_ref"] = pr

    if scol:
        raw["side"] = raw[scol].apply(_normalize_side)
    else:
        raw["side"] = np.where(
            pd.to_numeric(raw.get(qcol, 0), errors="coerce").fillna(0) >= 0,
            "BUY",
            "SELL",
        )

    out = raw[["date", "ticker", "qty", "px_ref", "side"]].copy()
    out = out.dropna()
    out = out[(out["qty"] > 0) & (out["px_ref"] > 0)]
    out["date"] = pd.to_datetime(out["date"]).dt.date
    return out


def _impact_bps(participation_pct: float) -> float:
    rel = max(participation_pct / 1.0, 1e-6)
    return BASE_IMPACT_BPS * (rel**IMPACT_EXPONENT)


def _schedule_slippage_bps(
    style: str, side: str, notional_inr: float, adv_inr: float
) -> float:
    # participation estimate per style
    if adv_inr <= 0:
        adv_inr = FALLBACK_ADV_INR
    pct = 100.0 * notional_inr / adv_inr
    if style == "POV":
        pct = min(POV_TARGET_PCT, PARTICIPATION_CAP_PCT, pct)
    else:
        pct = min(pct, PARTICIPATION_CAP_PCT)

    slip = MIN_SLIPPAGE_BPS + _impact_bps(pct)
    if style == "VWAP":
        slip += VWAP_DRIFT_BPS if side == "BUY" else -VWAP_DRIFT_BPS
    # TWAP drift ≈ 0; POV inherits only impact
    return slip


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)

    orders = _load_orders()
    adv_map = _lookup_adv_map()

    rows = []
    styles = ["TWAP", "VWAP", "POV"]
    for _, r in orders.iterrows():
        date = r["date"]
        tic = r["ticker"]
        side = r["side"]
        qty = int(r["qty"])
        px = float(r["px_ref"])
        notional = abs(qty) * px
        adv_inr = adv_map.get(tic, FALLBACK_ADV_INR)

        for style in styles:
            slip_bps = _schedule_slippage_bps(style, side, notional, adv_inr)
            sign = 1 if side == "BUY" else -1
            px_fill = px + (_bps(slip_bps) * px * sign)
            slip_inr = abs(_bps(slip_bps) * notional)
            comm_inr = abs(_bps(DEFAULT_COMMISSION_BPS) * notional)
            tax_inr = abs(_bps(DEFAULT_TAX_BPS) * notional)
            tca_inr = slip_inr + comm_inr + tax_inr
            part_pct = min(
                100.0 * notional / (adv_inr if adv_inr > 0 else FALLBACK_ADV_INR),
                PARTICIPATION_CAP_PCT,
            )

            rows.append(
                {
                    "date": date,
                    "ticker": tic,
                    "side": side,
                    "style": style,
                    "qty": qty,
                    "px_ref": round(px, 6),
                    "px_fill": round(px_fill, 6),
                    "notional_ref": round(notional, 2),
                    "slippage_bps": round(slip_bps, 3),
                    "participation_pct": round(part_pct, 3),
                    "adv_inr": round(float(adv_inr), 2),
                    "slip_cost_inr": round(slip_inr, 2),
                    "comm_inr": round(comm_inr, 2),
                    "tax_inr": round(tax_inr, 2),
                    "tca_cost_inr": round(tca_inr, 2),
                }
            )

    detail = pd.DataFrame(rows)
    if detail.empty:
        detail = pd.DataFrame(
            columns=[
                "date",
                "ticker",
                "side",
                "style",
                "qty",
                "px_ref",
                "px_fill",
                "notional_ref",
                "slippage_bps",
                "participation_pct",
                "adv_inr",
                "slip_cost_inr",
                "comm_inr",
                "tax_inr",
                "tca_cost_inr",
            ]
        )
    detail.to_csv(DETAIL_CSV, index=False)

    # Style-level rollup
    if not detail.empty:
        style_sum = detail.groupby("style", as_index=False).agg(
            orders=("ticker", "count"),
            notional_inr=("notional_ref", "sum"),
            tca_cost_inr=("tca_cost_inr", "sum"),
            med_slip_bps=("slippage_bps", "median"),
            p90_slip_bps=("slippage_bps", lambda s: float(np.percentile(s, 90))),
        )
    else:
        style_sum = pd.DataFrame(
            columns=[
                "style",
                "orders",
                "notional_inr",
                "tca_cost_inr",
                "med_slip_bps",
                "p90_slip_bps",
            ]
        )
    style_sum.to_csv(STYLE_SUMMARY_CSV, index=False)

    # Choose winner per order (min tca_cost)
    if not detail.empty:
        idx = detail.groupby(["date", "ticker"], as_index=False)[
            "tca_cost_inr"
        ].idxmin()
        winners = detail.loc[idx["tca_cost_inr"]].copy()
    else:
        winners = pd.DataFrame(columns=detail.columns)

    # Portfolio totals
    tot_orders = int(detail.shape[0])
    tot_notional = float(detail["notional_ref"].sum()) if not detail.empty else 0.0
    # Also provide best-style summary
    best_tca = float(winners["tca_cost_inr"].sum()) if not winners.empty else 0.0

    # High-level CSV (tiny)
    pd.DataFrame(
        [
            {
                "detail_csv": str(DETAIL_CSV),
                "style_summary_csv": str(STYLE_SUMMARY_CSV),
                "orders_evaluated": tot_orders,
                "notional_inr": round(tot_notional, 2),
                "best_total_tca_inr": round(best_tca, 2),
                "best_style_portfolio_hint": (
                    winners["style"].mode().iat[0] if not winners.empty else "NA"
                ),
            }
        ]
    ).to_csv(SUMMARY_CSV, index=False)

    # Console JSON
    out = {
        "detail_csv": str(DETAIL_CSV),
        "style_summary_csv": str(STYLE_SUMMARY_CSV),
        "summary_csv": str(SUMMARY_CSV),
        "orders_evaluated": tot_orders,
        "notional_inr": round(tot_notional, 2),
        "best_total_tca_inr": round(best_tca, 2),
        "best_style_portfolio_hint": (
            winners["style"].mode().iat[0] if not winners.empty else "NA"
        ),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

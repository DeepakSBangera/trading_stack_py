# scripts/w15_broker_sim_pov.py
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"F:\Projects\trading_stack_py")
REPORTS = ROOT / "reports"
DATA = ROOT / "data" / "prices"

FILLS_CSV = (
    REPORTS / "wk13_dryrun_fills.csv"
)  # inputs: qty, side, px_ref, notional_ref, etc.
ADV_PARQUET = REPORTS / "adv_value.parquet"  # optional: columns (ticker, adv_value) INR
CURVES_CSV = REPORTS / "w15_exec_curves_by_ticker.csv"
SUMMARY_CSV = REPORTS / "w15_exec_summary.csv"
DIAG_JSON = REPORTS / "w15_broker_sim_diag.json"

# --- ladder & cost knobs (align with W13 defaults) ---
POV_LADDER_PCT = [2.5, 5, 7.5, 10, 12.5, 15, 20, 25, 30, 40, 50]
PARTICIPATION_CAP_PCT = 50.0  # hard cap
MIN_SLIPPAGE_BPS = 2.0  # floor microstructure + spread
BASE_IMPACT_BPS = 8.0  # base impact at 1x rel participation
IMPACT_EXPONENT = 0.65  # concavity
VWAP_DRIFT_BPS = 3.0  # intraday drift term (buy positive, sell negative)
DEFAULT_COMMISSION_BPS = 1.5
DEFAULT_TAX_BPS = 0.0
FALLBACK_ADV_INR = 5e7  # if ADV missing

DATE_CANDS = ["date", "dt", "trading_day", "asof", "as_of"]
TICK_CANDS = ["ticker", "symbol", "name"]
QTY_CANDS = ["qty", "quantity", "shares", "qty_abs", "order_qty"]
SIDE_CANDS = ["side", "action", "direction", "buy_sell"]
PX_CANDS = ["px_ref", "ref_price", "price", "px_close", "close", "last", "close_price"]


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


def _norm_side(v: str | None) -> int:
    if not isinstance(v, str):
        return 0
    u = v.strip().upper()
    if u in ("B", "BUY", "LONG", "BUY_TO_OPEN", "OPEN_BUY"):
        return +1
    if u in ("S", "SELL", "SHORT", "SELL_TO_CLOSE", "CLOSE_SELL"):
        return -1
    return 0


def _bps(x: float) -> float:
    return x / 10000.0


def _load_adv_map() -> dict[str, float]:
    try:
        if ADV_PARQUET.exists():
            adv = pd.read_parquet(ADV_PARQUET)
            t = _pick(adv.columns, ["ticker", "symbol", "name"])
            a = _pick(adv.columns, ["adv_value", "adv_inr", "adv"])
            if t and a:
                ser = pd.to_numeric(adv[a], errors="coerce").fillna(0.0)
                return dict(zip(adv[t].astype(str), ser.astype(float), strict=False))
    except Exception:
        pass
    return {}


def _load_fills_lastday():
    if not FILLS_CSV.exists():
        raise SystemExit(f"Missing {FILLS_CSV}. Run W13 first.")
    df = pd.read_csv(FILLS_CSV)
    d = _pick(df.columns, DATE_CANDS) or "date"
    t = _pick(df.columns, TICK_CANDS) or "ticker"
    q = _pick(df.columns, QTY_CANDS) or "qty"
    s = _pick(df.columns, SIDE_CANDS) or "side"
    p = _pick(df.columns, PX_CANDS) or "px_ref"

    if not {d, t, q, p}.issubset(df.columns):
        raise SystemExit("fills missing date/ticker/qty/px_ref-like columns")

    df[d] = pd.to_datetime(df[d], errors="coerce").dt.date
    last = df[d].max()

    df["qty"] = pd.to_numeric(df[q], errors="coerce").fillna(0).astype(int)
    df["side_sign"] = (
        df[s].apply(_norm_side) if s in df.columns else np.sign(df["qty"]).astype(int)
    )
    df["px_ref"] = pd.to_numeric(df[p], errors="coerce").fillna(0.0).astype(float)
    df = df[df[d] == last].copy()

    # signed notional at reference price
    df["notional_ref"] = df["px_ref"].abs() * df["qty"].abs()
    # collapse by ticker, keep side by net signed qty
    g = df.groupby(t, as_index=False).agg(
        qty=("qty", "sum"),
        side_side=("side_sign", "sum"),
        px_ref=("px_ref", "median"),
        notional_ref=("notional_ref", "sum"),
    )
    g.rename(columns={t: "ticker"}, inplace=True)
    g["side_sign"] = np.sign(g["side_side"]).astype(int).replace(0, 1)  # neutral → buy
    return last, g[["ticker", "qty", "side_sign", "px_ref", "notional_ref"]].copy()


def _cost_for_order(
    notional_inr: float, pov_pct: float, side_sign: int, adv_inr: float
) -> tuple[float, float, float]:
    if notional_inr <= 0 or adv_inr <= 0:
        return (0.0, 0.0, 0.0)
    effective_pov = min(float(pov_pct), PARTICIPATION_CAP_PCT)
    rel = max(effective_pov / 1.0, 1e-6)  # normalized
    impact_bps = BASE_IMPACT_BPS * (rel**IMPACT_EXPONENT)
    slippage_bps = (
        MIN_SLIPPAGE_BPS + impact_bps + (VWAP_DRIFT_BPS * (1 if side_sign > 0 else -1))
    )
    comm_bps = DEFAULT_COMMISSION_BPS
    tax_bps = DEFAULT_TAX_BPS
    tca_bps = slippage_bps + comm_bps + tax_bps
    tca_inr = abs(_bps(tca_bps) * notional_inr)
    return (slippage_bps, tca_bps, tca_inr)


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    last_day, orders = _load_fills_lastday()
    adv_map = _load_adv_map()

    curve_rows = []
    for _, r in orders.iterrows():
        tic = str(r["ticker"])
        q = int(r["qty"])
        side = int(r["side_sign"])
        px = float(r["px_ref"])
        notl = float(r["notional_ref"])
        adv = float(adv_map.get(tic, FALLBACK_ADV_INR))

        for pov in POV_LADDER_PCT:
            slp_bps, tca_bps, tca_inr = _cost_for_order(notl, pov, side, adv)
            curve_rows.append(
                {
                    "ticker": tic,
                    "pov_pct": pov,
                    "orders": 1,
                    "notional_inr": round(notl, 2),
                    "slippage_bps": round(slp_bps, 4),
                    "tca_bps": round(tca_bps, 4),
                    "tca_inr": round(tca_inr, 2),
                }
            )

    curves = pd.DataFrame(curve_rows)
    if curves.empty:
        # emit empty shells for robustness
        curves = pd.DataFrame(
            columns=[
                "ticker",
                "pov_pct",
                "orders",
                "notional_inr",
                "slippage_bps",
                "tca_bps",
                "tca_inr",
            ]
        )

    # aggregate per (ticker, pov)
    agg = curves.groupby(["ticker", "pov_pct"], as_index=False).agg(
        orders=("orders", "sum"),
        notional_inr=("notional_inr", "sum"),
        med_slip_bps=("slippage_bps", "median"),
        p90_slip_bps=(
            "slippage_bps",
            lambda s: float(np.percentile(s, 90)) if len(s) else 0.0,
        ),
        exp_tca_inr=("tca_inr", "sum"),
    )
    agg["exp_tca_bps"] = np.where(
        agg["notional_inr"] > 0,
        (agg["exp_tca_inr"] / agg["notional_inr"]) * 10000.0,
        0.0,
    )

    # choose best POV per ticker by expected INR cost
    best_rows = []
    for tic, sub in agg.groupby("ticker"):
        sub = sub.sort_values("exp_tca_inr")
        best = sub.iloc[0]
        best_rows.append(
            {
                "ticker": tic,
                "best_pov_pct": float(best["pov_pct"]),
                "best_exp_tca_inr": float(best["exp_tca_inr"]),
                "best_exp_tca_bps": float(best["exp_tca_bps"]),
                "best_med_slip_bps": float(best["med_slip_bps"]),
            }
        )
    best = pd.DataFrame(best_rows)

    # overall POV frontier (portfolio)
    port = agg.groupby("pov_pct", as_index=False).agg(
        total_notional=("notional_inr", "sum"),
        exp_tca_inr=("exp_tca_inr", "sum"),
        med_slip_bps=("med_slip_bps", "median"),
        p90_slip_bps=("p90_slip_bps", "median"),
    )
    port["exp_tca_bps"] = np.where(
        port["total_notional"] > 0,
        (port["exp_tca_inr"] / port["total_notional"]) * 10000.0,
        0.0,
    )
    port_best = port.sort_values("exp_tca_inr").iloc[0] if not port.empty else None

    # write outputs
    agg.to_csv(CURVES_CSV, index=False)
    summary = best.copy()
    if port_best is not None:
        summary.loc[len(summary)] = {
            "ticker": "__PORTFOLIO__",
            "best_pov_pct": float(port_best["pov_pct"]),
            "best_exp_tca_inr": float(port_best["exp_tca_inr"]),
            "best_exp_tca_bps": float(port_best["exp_tca_bps"]),
            "best_med_slip_bps": float(port_best["med_slip_bps"]),
        }
    summary.to_csv(SUMMARY_CSV, index=False)

    DIAG_JSON.write_text(
        json.dumps(
            {
                "fills_day": str(last_day),
                "tickers": int(orders.shape[0]),
                "pov_grid": POV_LADDER_PCT,
                "curves_csv": str(CURVES_CSV),
                "summary_csv": str(SUMMARY_CSV),
                "adv_source": (
                    str(ADV_PARQUET)
                    if ADV_PARQUET.exists()
                    else f"fallback:{FALLBACK_ADV_INR} INR"
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # console summary
    out = {
        "curves_csv": str(CURVES_CSV),
        "summary_csv": str(SUMMARY_CSV),
        "tickers": int(orders.shape[0]),
        "portfolio_best_pov": (
            float(port_best["pov_pct"]) if port_best is not None else None
        ),
        "portfolio_exp_tca_bps": (
            float(port_best["exp_tca_bps"]) if port_best is not None else None
        ),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

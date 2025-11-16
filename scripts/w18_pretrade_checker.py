# scripts/w18_pretrade_checker.py
from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"F:\Projects\trading_stack_py")
REPORTS = ROOT / "reports"

# Inputs (produced earlier in your pipeline)
ORDERS_CSV = REPORTS / "wk12_orders_lastday.csv"  # v3 emitted with px_ref/qty/side
EVENT_FLAGS_CSV = REPORTS / "events_position_flags.csv"  # from W8
RISK_SCHEDULE_CSV = REPORTS / "risk_schedule_blended.csv"  # from W8 combine
TARGETS_CSV = REPORTS / "wk11_blend_targets.csv"  # optional (allow_new/rebalance)
PRICES_DIR = ROOT / "data" / "prices"  # parquet fallback for sanity

# Outputs
VIOLATIONS_CSV = REPORTS / "w18_pretrade_violations.csv"
OK_CSV = REPORTS / "w18_pretrade_ok.csv"
SUMMARY_JSON = REPORTS / "w18_pretrade_summary.json"

# knobs
REQ_COLS_ORDERS_ANY = {
    "date": ["date", "trading_day", "dt"],
    "ticker": ["ticker", "symbol", "name"],
    "side": ["side", "action", "direction", "buy_sell"],
    "qty": ["qty", "quantity", "shares", "shares_delta", "delta_qty", "order_qty"],
    "px_ref": [
        "px_ref",
        "ref_price",
        "price",
        "px_close",
        "close",
        "last",
        "close_price",
        "px",
    ],
}
SIDE_BUY = {"B", "BUY", "LONG", "BUY_TO_OPEN", "BUY TO OPEN", "OPEN_BUY"}
SIDE_SELL = {"S", "SELL", "SHORT", "SELL_TO_CLOSE", "SELL TO CLOSE", "CLOSE_SELL"}


def _pick_col(df: pd.DataFrame, cands: list[str]) -> str | None:
    low = {c.lower(): c for c in df.columns}
    for k in cands:
        if k in low:
            return low[k]
    norm = {c.lower().replace(" ", "").replace("-", "_"): c for c in df.columns}
    for k in cands:
        kk = k.replace(" ", "").replace("-", "_")
        if kk in norm:
            return norm[kk]
    return None


def _coerce_num(s, default=np.nan):
    return pd.to_numeric(s, errors="coerce").fillna(default)


def _norm_side(x: str) -> str | None:
    if not isinstance(x, str):
        return None
    u = x.strip().upper()
    if u in SIDE_BUY:
        return "BUY"
    if u in SIDE_SELL:
        return "SELL"
    return None


def _load_orders() -> pd.DataFrame:
    if not ORDERS_CSV.exists():
        raise FileNotFoundError(f"Missing {ORDERS_CSV}. Run W12 first.")
    df = pd.read_csv(ORDERS_CSV)
    dcol = _pick_col(df, REQ_COLS_ORDERS_ANY["date"])
    tcol = _pick_col(df, REQ_COLS_ORDERS_ANY["ticker"])
    scol = _pick_col(df, REQ_COLS_ORDERS_ANY["side"])
    qcol = _pick_col(df, REQ_COLS_ORDERS_ANY["qty"])
    pcol = _pick_col(df, REQ_COLS_ORDERS_ANY["px_ref"])
    if not all([dcol, tcol, qcol, pcol]):
        raise ValueError(
            f"orders missing required columns; got date={dcol}, ticker={tcol}, qty={qcol}, px={pcol}"
        )

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df[dcol], errors="coerce").dt.date,
            "ticker": df[tcol].astype(str),
            "side_raw": df[scol] if scol else "",
            "qty": _coerce_num(df[qcol], 0).astype(float),
            "px_ref": _coerce_num(df[pcol], np.nan).astype(float),
        }
    )
    out["side"] = out["side_raw"].apply(_norm_side)
    # infer side by qty sign if missing
    out.loc[out["side"].isna() & (out["qty"] > 0), "side"] = "BUY"
    out.loc[out["side"].isna() & (out["qty"] < 0), "side"] = "SELL"
    out["qty_abs"] = out["qty"].abs()
    # drop rows with no date/ticker
    out = out[out["date"].notna() & out["ticker"].notna()]
    return out


def _load_event_flags() -> pd.DataFrame:
    if not EVENT_FLAGS_CSV.exists():
        return pd.DataFrame(columns=["date", "ticker", "do_not_trade"])
    ev = pd.read_csv(EVENT_FLAGS_CSV)
    dcol = _pick_col(ev, ["date", "dt"])
    tcol = _pick_col(ev, ["ticker", "symbol", "name"])
    fcol = _pick_col(ev, ["do_not_trade", "blocked", "halt", "skip"])
    if not all([dcol, tcol, fcol]):  # tolerate shape changes
        return pd.DataFrame(columns=["date", "ticker", "do_not_trade"])
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(ev[dcol], errors="coerce").dt.date,
            "ticker": ev[tcol].astype(str),
            "do_not_trade": ev[fcol]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(["1", "true", "yes", "y", "t"]),
        }
    )
    return out


def _load_risk_schedule() -> pd.DataFrame:
    if not RISK_SCHEDULE_CSV.exists():
        return pd.DataFrame(columns=["date", "ticker", "final_mult"])
    rs = pd.read_csv(RISK_SCHEDULE_CSV)
    dcol = _pick_col(rs, ["date", "dt"])
    tcol = _pick_col(rs, ["ticker", "symbol", "name"])
    mcol = _pick_col(rs, ["final_mult", "mult", "total_mult", "risk_mult"])
    if not all([dcol, tcol, mcol]):
        return pd.DataFrame(columns=["date", "ticker", "final_mult"])
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(rs[dcol], errors="coerce").dt.date,
            "ticker": rs[tcol].astype(str),
            "final_mult": _coerce_num(rs[mcol], 1.0).astype(float),
        }
    )
    return out


def _load_targets_lite() -> pd.DataFrame:
    if not TARGETS_CSV.exists():
        return pd.DataFrame(
            columns=["date", "ticker", "allow_new_final", "rebalance_allowed_final"]
        )
    tg = pd.read_csv(TARGETS_CSV)
    dcol = _pick_col(tg, ["date"])
    tcol = _pick_col(tg, ["ticker"])
    anew = _pick_col(tg, ["allow_new_final", "allow_new"])
    rbal = _pick_col(tg, ["rebalance_allowed_final", "rebalance_allowed"])
    if not all([dcol, tcol]):
        return pd.DataFrame(
            columns=["date", "ticker", "allow_new_final", "rebalance_allowed_final"]
        )
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(tg[dcol], errors="coerce").dt.date,
            "ticker": tg[tcol].astype(str),
            "allow_new_final": tg[anew].astype(bool) if anew else True,
            "rebalance_allowed_final": tg[rbal].astype(bool) if rbal else True,
        }
    )
    return out


def _check_row(r, ev_map, mult_map, allow_map):
    viols = []

    # Basic shape
    if not isinstance(r["date"], dt.date):
        viols.append(("shape", "invalid_date"))
    if not isinstance(r["ticker"], str):
        viols.append(("shape", "invalid_ticker"))
    if r["side"] not in ("BUY", "SELL"):
        viols.append(("shape", f"bad_side:{r['side']}"))
    if not math.isfinite(r["qty_abs"]) or r["qty_abs"] <= 0:
        viols.append(("shape", f"bad_qty:{r['qty_abs']}"))
    if not math.isfinite(r["px_ref"]) or r["px_ref"] <= 0:
        viols.append(("shape", f"bad_px:{r['px_ref']}"))

    key = (r["date"], r["ticker"])
    # Events
    if ev_map.get(key, False):
        viols.append(("events", "do_not_trade"))

    # Regime/risk multiplier
    mult = mult_map.get(key, 1.0)
    if mult <= 0.0:
        viols.append(("regime", f"blocked_mult:{mult}"))

    # Targets allow_new / rebalance gate (best-effort)
    allow = allow_map.get(key, (True, True))
    allow_new, reb_ok = allow
    # New position (qty>0 & side=BUY) without allow_new
    if r["side"] == "BUY" and not allow_new:
        viols.append(("targets", "new_not_allowed"))
    # Rebalance SELL to reduce within target might be blocked (we treat SELL when rebalance not allowed)
    if r["side"] == "SELL" and not reb_ok:
        viols.append(("targets", "rebalance_not_allowed"))

    return viols


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)

    orders = _load_orders()
    events = _load_event_flags()
    sched = _load_risk_schedule()
    targets = _load_targets_lite()

    # Build lookup maps
    ev_map = (
        {
            (d, t): f
            for d, t, f in zip(
                events["date"], events["ticker"], events["do_not_trade"], strict=False
            )
        }
        if not events.empty
        else {}
    )
    mult_map = (
        {
            (d, t): m
            for d, t, m in zip(
                sched["date"], sched["ticker"], sched["final_mult"], strict=False
            )
        }
        if not sched.empty
        else {}
    )
    allow_map = (
        {
            (d, t): (an, rb)
            for d, t, an, rb in zip(
                targets.get("date", []),
                targets.get("ticker", []),
                targets.get("allow_new_final", []),
                targets.get("rebalance_allowed_final", []),
                strict=False,
            )
        }
        if not targets.empty
        else {}
    )

    viol_rows = []
    ok_rows = []

    for _, r in orders.iterrows():
        viols = _check_row(r, ev_map, mult_map, allow_map)
        if viols:
            for kind, msg in viols:
                viol_rows.append(
                    {
                        "date": r["date"],
                        "ticker": r["ticker"],
                        "side": r["side"],
                        "qty": int(abs(r["qty"])),
                        "px_ref": float(r["px_ref"]),
                        "violation": kind,
                        "detail": msg,
                    }
                )
        else:
            ok_rows.append(
                {
                    "date": r["date"],
                    "ticker": r["ticker"],
                    "side": r["side"],
                    "qty": int(abs(r["qty"])),
                    "px_ref": float(r["px_ref"]),
                }
            )

    viol_df = pd.DataFrame(viol_rows)
    ok_df = pd.DataFrame(ok_rows)

    viol_df.to_csv(VIOLATIONS_CSV, index=False)
    ok_df.to_csv(OK_CSV, index=False)

    summary = {
        "orders_in": int(orders.shape[0]),
        "ok": int(ok_df.shape[0]),
        "violations": int(viol_df.shape[0]),
        "unique_tickers": int(orders["ticker"].nunique()),
        "files": {"ok_csv": str(OK_CSV), "violations_csv": str(VIOLATIONS_CSV)},
        "inputs_present": {
            "orders_csv": ORDERS_CSV.exists(),
            "event_flags_csv": EVENT_FLAGS_CSV.exists(),
            "risk_schedule_csv": RISK_SCHEDULE_CSV.exists(),
            "targets_csv": TARGETS_CSV.exists(),
        },
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

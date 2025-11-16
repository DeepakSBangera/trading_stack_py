# scripts/w14_t1_reco.py
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
DOCS = ROOT / "docs"

FILLS_CSV = REPORTS / "wk13_dryrun_fills.csv"  # from W13
TARGETS_CSV = REPORTS / "wk11_blend_targets.csv"  # W11 targets, by date
POS_IN_CSV = REPORTS / "positions_in.csv"  # optional starting positions snapshot
POS_OUT_CSV = REPORTS / "w14_positions_reco.csv"  # new positions after fills (T+1)
DRIFT_CSV = REPORTS / "w14_drift_summary.csv"  # drift vs next-day targets
DIAG_JSON = REPORTS / "w14_reco_diag.json"  # metadata

# If no POS_IN_CSV exists, we assume flat (0) coming into T
ASSUME_START_FLAT = True

DATE_CANDS = ["date", "dt", "trading_day", "asof", "as_of"]
TICK_CANDS = ["ticker", "symbol", "name"]
QTY_CANDS = ["qty", "quantity", "shares", "position", "pos", "qty_abs"]
SIDE_CANDS = ["side", "action", "buy_sell", "order_side", "direction"]


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


def _load_fills():
    if not FILLS_CSV.exists():
        raise SystemExit(f"Missing {FILLS_CSV}. Run W13 first.")
    f = pd.read_csv(FILLS_CSV)
    d = _pick(f.columns, DATE_CANDS) or "date"
    t = _pick(f.columns, TICK_CANDS) or "ticker"
    q = _pick(f.columns, QTY_CANDS) or "qty"
    s = _pick(f.columns, SIDE_CANDS) or "side"
    if not {d, t, q}.issubset(f.columns):
        raise SystemExit("fills missing date/ticker/qty columns")
    f[d] = pd.to_datetime(f[d], errors="coerce").dt.date
    f["qty_int"] = pd.to_numeric(f[q], errors="coerce").fillna(0).astype(int)
    if s in f.columns:
        f["sign"] = f[s].apply(_norm_side)
        # some files already have signed qty; if side unknown, use qty sign
        f.loc[f["sign"] == 0, "sign"] = np.sign(f["qty_int"]).astype(int)
    else:
        f["sign"] = np.sign(f["qty_int"]).astype(int)
    f["dq"] = f["qty_int"] * f["sign"]
    # collapse to last day
    last = f[d].max()
    f = f[f[d] == last].copy()
    f = f.groupby([t], as_index=False)["dq"].sum()
    f.rename(columns={t: "ticker"}, inplace=True)
    return last, f


def _load_targets():
    if not TARGETS_CSV.exists():
        raise SystemExit(f"Missing {TARGETS_CSV}. Run W11 first.")
    df = pd.read_csv(TARGETS_CSV)
    d = _pick(df.columns, DATE_CANDS)
    t = _pick(df.columns, TICK_CANDS)
    w = _pick(df.columns, ["target_w", "weight", "w", "blend_w", "final_w"])
    if not d or not t or not w:
        raise SystemExit("wk11_blend_targets.csv missing date/ticker/weight")
    df[d] = pd.to_datetime(df[d], errors="coerce").dt.date
    # we need last day (T) and next trading day (T+1 in file)
    all_days = sorted(set(df[d].dropna()))
    return df.rename(columns={d: "date", t: "ticker", w: "target_w"}), all_days


def _load_start_positions(T_date: dt.date, universe: list[str]) -> pd.DataFrame:
    # Try POS_IN_CSV → else flat
    if POS_IN_CSV.exists():
        p = pd.read_csv(POS_IN_CSV)
        t = _pick(p.columns, TICK_CANDS)
        q = _pick(p.columns, QTY_CANDS)
        if t and q:
            p = p[[t, q]].copy()
            p.columns = ["ticker", "pos_T"]
            p["ticker"] = p["ticker"].astype(str)
            return p
    if ASSUME_START_FLAT:
        return pd.DataFrame({"ticker": universe, "pos_T": 0}, dtype=object)
    raise SystemExit("No starting positions available (and ASSUME_START_FLAT=False).")


def _lookup_close(ticker: str, date_val: dt.date) -> float | None:
    p = DATA / f"{ticker}.parquet"
    if not p.exists():
        return None
    try:
        x = pd.read_parquet(p)
        d = _pick(x.columns, ["date", "dt"])
        c = _pick(x.columns, ["close", "px_close", "price"])
        if not d or not c:
            return None
        x[d] = pd.to_datetime(x[d], errors="coerce")
        x = x.dropna(subset=[d]).sort_values(d)
        tgt = pd.to_datetime(date_val)
        m = x[x[d] == tgt]
        if not m.empty:
            v = float(pd.to_numeric(m.iloc[0][c], errors="coerce"))
            return v if math.isfinite(v) and v > 0 else None
        prev = x[x[d] < tgt]
        if not prev.empty:
            v = float(pd.to_numeric(prev.iloc[-1][c], errors="coerce"))
            return v if math.isfinite(v) and v > 0 else None
    except Exception:
        return None
    return None


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    targets, days = _load_targets()
    if not days:
        raise SystemExit("No dates in targets.")
    T = max(days)  # using the latest day in targets file as T
    T1_candidates = [d for d in days if d > T]
    T1 = (
        T1_candidates[0] if T1_candidates else None
    )  # often None in rolling sims; we’ll still compute drift vs T targets
    last_fills_day, fills = _load_fills()
    if last_fills_day != T:
        # still OK—use fills from their last day to apply on T (best-effort)
        pass

    # Universe on T (from targets)
    univ_T = targets.loc[targets["date"] == T, "ticker"].astype(str).unique().tolist()
    pos_T = _load_start_positions(T, univ_T)

    # Apply fills to get T+1 positions
    pos = pos_T.merge(fills, how="outer", left_on="ticker", right_on="ticker")
    pos["pos_T"] = pd.to_numeric(pos["pos_T"], errors="coerce").fillna(0).astype(int)
    pos["dq"] = pd.to_numeric(pos["dq"], errors="coerce").fillna(0).astype(int)
    pos["pos_T1"] = (pos["pos_T"] + pos["dq"]).astype(int)

    # Notional estimates using T closes (best-effort if available)
    pos["px_T"] = [_lookup_close(t, T) for t in pos["ticker"].astype(str)]
    pos["notional_T1_inr"] = pos["pos_T1"].abs() * pd.to_numeric(
        pos["px_T"], errors="coerce"
    ).fillna(0.0)

    # Drift vs targets: if T1 exists, compare to targets on T1; else compare to targets on T
    dref = T1 if T1 is not None else T
    tgt = targets[targets["date"] == dref][["ticker", "target_w"]].copy()
    tgt["ticker"] = tgt["ticker"].astype(str)
    # compute portfolio notional proxy
    port_notional = float(pos["notional_T1_inr"].sum())
    pos = pos.merge(tgt, how="left", on="ticker")
    pos["w_realized"] = np.where(
        port_notional > 0.0, (pos["notional_T1_inr"] / port_notional), 0.0
    )
    pos["drift_w"] = pos["w_realized"].fillna(0.0) - pos["target_w"].fillna(0.0)

    # Outputs
    POS_OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pos_out = pos[
        [
            "ticker",
            "pos_T",
            "dq",
            "pos_T1",
            "px_T",
            "notional_T1_inr",
            "target_w",
            "w_realized",
            "drift_w",
        ]
    ].copy()
    pos_out = pos_out.sort_values("ticker")
    pos_out.to_csv(POS_OUT_CSV, index=False)

    # Drift summary
    drift = {
        "date_T": str(T),
        "date_ref_for_targets": str(dref),
        "names": int(pos_out.shape[0]),
        "gross_notional_inr": float(port_notional),
        "abs_drift_sum": float(pos_out["drift_w"].abs().sum()),
        "abs_drift_p90": (
            float(np.percentile(pos_out["drift_w"].abs(), 90))
            if pos_out.shape[0]
            else 0.0
        ),
        "top5_by_abs_drift": ";".join(
            pos_out.reindex(pos_out["drift_w"].abs().sort_values(ascending=False).index)
            .head(5)
            .apply(lambda r: f"{r['ticker']}:{round(float(r['drift_w']), 6)}", axis=1)
            .tolist()
        ),
    }
    pd.DataFrame([drift]).to_csv(DRIFT_CSV, index=False)

    DIAG_JSON.write_text(
        json.dumps(
            {
                "fills_day": str(last_fills_day),
                "T": str(T),
                "T1": str(T1),
                "pos_csv": str(POS_OUT_CSV),
                "drift_csv": str(DRIFT_CSV),
                "start_flat": ASSUME_START_FLAT and (not POS_IN_CSV.exists()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "pos_csv": str(POS_OUT_CSV),
                "drift_csv": str(DRIFT_CSV),
                "names": int(pos_out.shape[0]),
                "abs_drift_sum": round(drift["abs_drift_sum"], 8),
                "abs_drift_p90": round(drift["abs_drift_p90"], 8),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(r"F:\Projects\trading_stack_py")
REPORTS = ROOT / "reports"
OUT_CSV = REPORTS / "wk11_blend_targets.csv"

SCHED = REPORTS / "risk_schedule_blended.csv"  # from W8 combiner
W6COMP = REPORTS / "wk6_portfolio_compare.csv"  # w_final baseline
POSPQ = REPORTS / "positions_daily.parquet"  # fallback baseline


def open_win(p: Path):
    if sys.platform.startswith("win") and p.exists():
        os.startfile(p)


def load_baseline_weights() -> pd.Series:
    # Prefer W6 capped weights (w_final); else last day from positions
    if W6COMP.exists():
        comp = pd.read_csv(W6COMP)
        if {"ticker", "w_final"}.issubset(comp.columns) and comp["w_final"].sum() > 0:
            w = comp.set_index("ticker")["w_final"].clip(lower=0.0)
            if w.sum() > 0:
                return w / w.sum()
    if POSPQ.exists():
        pos = pd.read_parquet(POSPQ)
        pos["date"] = pd.to_datetime(pos["date"])
        d = pos["date"].max()
        snap = pos[pos["date"] == d]
        if {"ticker", "weight"}.issubset(snap.columns) and snap["weight"].abs().sum() > 0:
            w = snap.set_index("ticker")["weight"].clip(lower=0.0)
            return w / w.sum() if w.sum() > 0 else w
    raise SystemExit("No baseline weights found (need wk6_portfolio_compare.csv or positions_daily.parquet).")


def renorm_with_locks(target: pd.Series, lock_mask: pd.Series, locked_values: pd.Series) -> pd.Series:
    """Keep locked tickers at locked_values; renormalize the rest to sum to (1 - sum_locked)."""
    target = target.copy()
    locked_sum = float(locked_values[lock_mask].sum()) if lock_mask.any() else 0.0
    free_sum = float(target[~lock_mask].sum())
    if free_sum <= 0:
        # Nothing free to scale; return locked only (rest zero)
        target[~lock_mask] = 0.0
        # if locked sum > 0, renorm to 1; else zero portfolio
        if locked_sum > 0:
            return (target / locked_sum).fillna(0.0)
        return target.fillna(0.0)
    scale = max(1e-12, (1.0 - locked_sum)) / free_sum
    target[~lock_mask] = target[~lock_mask] * scale
    # ensure exact 1.0 sum (tiny numeric fix)
    s = float(target.sum())
    if s > 0:
        target = target / s
    return target


def build_targets():
    if not SCHED.exists():
        raise SystemExit("Missing reports\\risk_schedule_blended.csv — run w8_combine_schedules.py first.")
    sched = pd.read_csv(SCHED, parse_dates=["date"])
    base_w = load_baseline_weights()

    # Ensure schedule covers exactly those tickers; if schedule has more, restrict; if less, add zeros
    tickers = sorted(set(base_w.index).intersection(set(sched["ticker"].unique())))
    if not tickers:
        # If no intersection (shouldn't happen), use schedule tickers & reindex base to 0
        tickers = sorted(sched["ticker"].unique())
        base_w = pd.Series(0.0, index=tickers)

    # pivot schedule by date × ticker for required columns
    need_cols = [
        "date",
        "ticker",
        "final_risk_multiplier",
        "allow_new_final",
        "rebalance_allowed_final",
    ]
    missing = [c for c in need_cols if c not in sched.columns]
    if missing:
        raise SystemExit(f"risk_schedule_blended.csv missing columns: {missing}")

    sched = sched[need_cols].copy()
    # Sort by date/ticker ensure stable iteration
    sched = sched.sort_values(["date", "ticker"])
    dates = list(pd.to_datetime(sched["date"].unique()))

    # State: previous day target weights (start from base_w)
    prev_w = base_w.reindex(tickers).fillna(0.0)
    if prev_w.sum() > 0:
        prev_w = prev_w / prev_w.sum()

    rows = []
    for d in dates:
        day = sched[sched["date"] == d].set_index("ticker").reindex(tickers)
        # desired scaled gross = base_w * multiplier
        mult = day["final_risk_multiplier"].astype(float).fillna(1.0)
        allow_new = day["allow_new_final"].fillna(True).astype(bool)
        rebalance_ok = day["rebalance_allowed_final"].fillna(True).astype(bool)

        desired = base_w * mult
        desired = desired.clip(lower=0.0)

        if not bool(rebalance_ok.all()):
            # If any rebalance is blocked, we lock all tickers where rebalance is False at prev weight
            # and only adjust the others proportionally to desired
            lock_mask = ~rebalance_ok  # rebalance not allowed
            locked_vals = prev_w.copy()
            # For allow_new=False and prev_w == 0 → force remain 0 (already handled by locked prev_w==0)
            unlocked_target = desired.copy()
            target = renorm_with_locks(unlocked_target, lock_mask, locked_vals)
        else:
            # Rebalance allowed globally; but enforce allow_new=False for names with zero prev weight
            locked_new_mask = (~allow_new) & (prev_w <= 1e-12)
            locked_vals = prev_w.where(locked_new_mask, 0.0)
            target = renorm_with_locks(desired, locked_new_mask, locked_vals)

        # Clean tiny negatives/NaNs
        target = target.fillna(0.0).clip(lower=0.0)
        if target.sum() > 0:
            target = target / target.sum()

        # Record rows
        for t in tickers:
            rows.append(
                {
                    "date": d.date().isoformat(),
                    "ticker": t,
                    "base_w": float(round(base_w.get(t, 0.0), 10)),
                    "final_mult": float(round(mult.get(t, 1.0), 6)),
                    "allow_new_final": bool(allow_new.get(t, True)),
                    "rebalance_allowed_final": bool(rebalance_ok.get(t, True)),
                    "target_w": float(round(target.get(t, 0.0), 10)),
                }
            )

        prev_w = target.copy()

    out = pd.DataFrame(rows).sort_values(["date", "ticker"])
    REPORTS.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    # Summaries
    by_date = out.groupby("date")["target_w"].sum().describe()
    info = {
        "rows": int(out.shape[0]),
        "dates": int(out["date"].nunique()),
        "tickers": int(out["ticker"].nunique()),
        "sum_target_w_min_max": [
            float(out.groupby("date")["target_w"].sum().min()),
            float(out.groupby("date")["target_w"].sum().max()),
        ],
    }
    print(json.dumps({"out_csv": str(OUT_CSV), **info}, indent=2))
    open_win(OUT_CSV)


if __name__ == "__main__":
    build_targets()

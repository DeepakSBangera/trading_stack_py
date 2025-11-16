from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

"""
W35 — Turnover & Hysteresis Tuning (grid search)

Inputs
- reports/wk11_blend_targets.csv  (date,ticker,target_w, …)
- Optional: reports/adv_value.parquet (ticker, adv_value)  — INR ADV per name

Method
- For each parameter set (entry_pp, exit_pp, min_hold_days):
  * We look at day-over-day target_w deltas per ticker.
  * A rebalance trade triggers only if |delta_w| > entry_pp (enter/raise) or < -exit_pp (trim/exit),
    with a simple “last trade day” hold constraint (min_hold_days).
  * Notional traded ≈ |delta_w| × DEFAULT_PORTFOLIO_NOTIONAL_INR.
  * Slippage bps = MIN_SLIPPAGE_BPS + BASE_IMPACT_BPS * ( participation ** IMPACT_EXPONENT )
    where participation = min( notional / ADV_INR, PARTICIPATION_CAP_PCT% )
    BUY adds +VWAP_DRIFT_BPS; SELL subtracts the same drift (optional bias).
  * Commission/Tax added in bps.

Outputs
- reports/wk35_tca_tuning.csv    (one row per parameter set; turnover & costs)
- reports/w35_tca_detail.csv     (optional: top candidates expanded)
- reports/w35_tca_summary.json   (chosen “recommendation” + knobs)
"""

ROOT = Path(r"F:\Projects\trading_stack_py")
REPORTS = ROOT / "reports"

TARGETS_CSV = REPORTS / "wk11_blend_targets.csv"
ADV_PARQUET = REPORTS / "adv_value.parquet"  # optional
OUT_GRID_CSV = REPORTS / "wk35_tca_tuning.csv"
OUT_DETAIL = REPORTS / "w35_tca_detail.csv"
OUT_SUMMARY = REPORTS / "w35_tca_summary.json"

# --- Portfolio & TCA knobs (align with W13 slippage model) ---
DEFAULT_PORTFOLIO_NOTIONAL_INR = 10_000_000  # ₹1 cr test size

DEFAULT_COMMISSION_BPS = 1.5
DEFAULT_TAX_BPS = 0.0
MIN_SLIPPAGE_BPS = 2.0
BASE_IMPACT_BPS = 8.0
IMPACT_EXPONENT = 0.65
PARTICIPATION_CAP_PCT = 12.5  # cap at 12.5% of ADV
VWAP_DRIFT_BPS = 3.0  # + for BUY, - for SELL

# --- Grid (feel free to tweak) ---
ENTRY_PPS = [0.0010, 0.0025, 0.0050, 0.0100]  # 0.10%, 0.25%, 0.50%, 1.00%
EXIT_PPS = [0.0005, 0.0010, 0.0025, 0.0050]  # 0.05%, 0.10%, 0.25%, 0.50%
HOLD_DAYS = [0, 3, 5]

RANDOM_SEED = 7


def _bps(x: float) -> float:
    return x / 10_000.0


def _load_adv_map() -> dict[str, float]:
    try:
        p = ADV_PARQUET
        if p.exists():
            adv = pd.read_parquet(p)
            cols = {c.lower(): c for c in adv.columns}
            tcol = cols.get("ticker")
            acol = cols.get("adv_value")
            if tcol and acol:
                return dict(
                    zip(
                        adv[tcol].astype(str),
                        pd.to_numeric(adv[acol], errors="coerce").fillna(0.0),
                        strict=False,
                    )
                )
    except Exception:
        pass
    return {}


def _load_targets() -> pd.DataFrame:
    if not TARGETS_CSV.exists():
        raise FileNotFoundError(f"Missing {TARGETS_CSV}; run W11 first.")
    df = pd.read_csv(TARGETS_CSV)
    cols = {c.lower(): c for c in df.columns}
    dcol = cols.get("date", "date")
    ticol = cols.get("ticker", "ticker")
    wcol = cols.get("target_w", "target_w")
    if dcol not in df.columns or ticol not in df.columns or wcol not in df.columns:
        raise ValueError(
            "wk11_blend_targets.csv must have columns: date,ticker,target_w"
        )
    df = df[[dcol, ticol, wcol]].rename(
        columns={dcol: "date", ticol: "ticker", wcol: "target_w"}
    )
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["ticker"] = df["ticker"].astype(str)
    df["target_w"] = pd.to_numeric(df["target_w"], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["date"])
    return df.sort_values(["ticker", "date"])


def _simulate_one(
    df: pd.DataFrame,
    entry_pp: float,
    exit_pp: float,
    min_hold_days: int,
    adv_map: dict[str, float],
) -> dict:
    """
    Return dict with metrics for one parameter set.
    """
    notional = DEFAULT_PORTFOLIO_NOTIONAL_INR
    rng = np.random.default_rng(RANDOM_SEED)

    trades = []
    last_trade_day_by_ticker: dict[str, pd.Timestamp] = {}

    # Iterate per ticker
    for tck, g in df.groupby("ticker"):
        g = g.sort_values("date")
        prev_w = None
        for _, row in g.iterrows():
            d = row["date"]
            w = float(row["target_w"])
            if prev_w is None:
                prev_w = w
                last_trade_day_by_ticker[tck] = d - pd.Timedelta(days=min_hold_days + 1)
                continue

            delta_w = w - prev_w
            trigger = False
            side = None
            if abs(delta_w) >= entry_pp:
                trigger = True
                side = "BUY" if delta_w > 0 else "SELL"
            elif abs(delta_w) >= exit_pp and (
                w == 0.0 or np.sign(delta_w) != np.sign(prev_w)
            ):
                # lighter exit trim (or sign flip)
                trigger = True
                side = "BUY" if delta_w > 0 else "SELL"

            # hold constraint
            last_d = last_trade_day_by_ticker.get(
                tck, d - pd.Timedelta(days=min_hold_days + 1)
            )
            if trigger and (d - last_d).days < min_hold_days:
                trigger = False  # skip due to hold

            if trigger:
                trade_notional = abs(delta_w) * notional
                adv_val = adv_map.get(tck, np.nan)
                if not math.isfinite(adv_val) or adv_val <= 0:
                    adv_val = 5e7  # 5 cr fallback ADV INR

                participation = min(
                    100.0 * trade_notional / adv_val, PARTICIPATION_CAP_PCT
                )
                rel = max(participation / 1.0, 1e-6)
                impact_bps = BASE_IMPACT_BPS * (rel**IMPACT_EXPONENT)
                slippage_bps = (
                    MIN_SLIPPAGE_BPS
                    + impact_bps
                    + (VWAP_DRIFT_BPS if side == "BUY" else -VWAP_DRIFT_BPS)
                )

                # Costs (in INR)
                slip_inr = abs(_bps(slippage_bps) * trade_notional)
                comm_inr = abs(_bps(DEFAULT_COMMISSION_BPS) * trade_notional)
                tax_inr = abs(_bps(DEFAULT_TAX_BPS) * trade_notional)

                trades.append(
                    {
                        "date": d.date(),
                        "ticker": tck,
                        "side": side,
                        "delta_w": delta_w,
                        "trade_notional": trade_notional,
                        "participation_pct": participation,
                        "slippage_bps": slippage_bps,
                        "slip_inr": slip_inr,
                        "comm_inr": comm_inr,
                        "tax_inr": tax_inr,
                        "tca_inr": slip_inr + comm_inr + tax_inr,
                    }
                )
                last_trade_day_by_ticker[tck] = d

            prev_w = w

    if len(trades) == 0:
        return {
            "entry_pp": entry_pp,
            "exit_pp": exit_pp,
            "min_hold_days": min_hold_days,
            "trades": 0,
            "turnover": 0.0,
            "notional": 0.0,
            "median_slip_bps": 0.0,
            "p90_slip_bps": 0.0,
            "slip_inr": 0.0,
            "comm_inr": 0.0,
            "tax_inr": 0.0,
            "tca_inr": 0.0,
        }

    td = pd.DataFrame(trades)
    turnover = float(td["delta_w"].abs().sum())
    notional_sum = float(td["trade_notional"].sum())
    med_slip = float(td["slippage_bps"].median())
    p90_slip = float(np.percentile(td["slippage_bps"], 90))
    slip_inr = float(td["slip_inr"].sum())
    comm_inr = float(td["comm_inr"].sum())
    tax_inr = float(td["tax_inr"].sum())
    tca_inr = float(td["tca_inr"].sum())

    return {
        "entry_pp": entry_pp,
        "exit_pp": exit_pp,
        "min_hold_days": min_hold_days,
        "trades": int(td.shape[0]),
        "turnover": turnover,
        "notional": notional_sum,
        "median_slip_bps": med_slip,
        "p90_slip_bps": p90_slip,
        "slip_inr": slip_inr,
        "comm_inr": comm_inr,
        "tax_inr": tax_inr,
        "tca_inr": tca_inr,
    }


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    adv_map = _load_adv_map()
    targets = _load_targets()

    # Ensure clean panel (date x ticker)
    targets = targets.dropna(subset=["target_w"])
    # Normalize to ensure weights sum ~1 per date (not strictly required for delta, but keeps scale tidy)
    sums = targets.groupby("date")["target_w"].sum().replace(0, np.nan)
    targets = targets.merge(sums.rename("w_sum"), on="date", how="left")
    targets["target_w"] = np.where(
        targets["w_sum"].abs() > 0,
        targets["target_w"] / targets["w_sum"],
        targets["target_w"],
    )
    targets = targets.drop(columns=["w_sum"])

    # Grid
    results = []
    for epp in ENTRY_PPS:
        for xpp in EXIT_PPS:
            for hd in HOLD_DAYS:
                met = _simulate_one(targets, epp, xpp, hd, adv_map)
                results.append(met)

    grid = pd.DataFrame(results).sort_values(["tca_inr", "turnover", "trades"])
    # Simple recommendation: minimize TCA subject to reasonable turnover
    # We can set an automatic turnover guard (e.g., <= 1.2 × median turnover across grid)
    turn_median = float(grid["turnover"].median()) if not grid.empty else 0.0
    if not grid.empty:
        cand = grid[grid["turnover"] <= (1.20 * max(turn_median, 1e-9))].copy()
        if cand.empty:
            cand = grid.copy()
        # Choose the smallest TCA in candidate set
        best_idx = cand["tca_inr"].idxmin()
        grid["recommend"] = False
        grid.loc[best_idx, "recommend"] = True
        best = grid.loc[best_idx].to_dict()
    else:
        grid["recommend"] = False
        best = None

    # Save outputs
    grid.to_csv(OUT_GRID_CSV, index=False)

    # Detail: top-10 candidates
    top = grid.nsmallest(10, ["tca_inr", "turnover", "trades"])
    top.to_csv(OUT_DETAIL, index=False)

    summary = {
        "rows": int(grid.shape[0]),
        "best": best,
        "defaults": {
            "portfolio_notional_inr": DEFAULT_PORTFOLIO_NOTIONAL_INR,
            "commission_bps": DEFAULT_COMMISSION_BPS,
            "tax_bps": DEFAULT_TAX_BPS,
            "min_slippage_bps": MIN_SLIPPAGE_BPS,
            "base_impact_bps": BASE_IMPACT_BPS,
            "impact_exponent": IMPACT_EXPONENT,
            "participation_cap_pct": PARTICIPATION_CAP_PCT,
            "vwap_drift_bps": VWAP_DRIFT_BPS,
        },
        "files": {"grid_csv": str(OUT_GRID_CSV), "detail_csv": str(OUT_DETAIL)},
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"grid_csv": str(OUT_GRID_CSV), "best": best}, indent=2))


if __name__ == "__main__":
    main()

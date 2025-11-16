from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

"""
W38 — Turnover Optimization Sprint

Goal
----
Quantify turnover and cost drag under different *rebalance cadence* and *hysteresis*
bands, then recommend the lowest-drag setting that still tracks the baseline closely.

Inputs
------
- reports/wk11_blend_targets.csv   -> provides the universe (tickers), must include columns: date, ticker
- data/prices/<TICKER>.parquet     -> must contain columns: date + close (or px_close/price)

Outputs
-------
- reports/wk38_turnover_profile.csv
    Columns: setting, rebalance, band_bps, annual_turnover, cost_drag_bps,
             tracking_corr, tracking_rmse_bps, notes
- reports/wk38_turnover_reco.json
    { "recommended": { ...row fields... }, "tested": <n>, "window_days": <n> }

Method (self-contained)
-----------------------
1) Build panel of closes for the W11 universe.
2) Score = 63d momentum (px/px[-63]-1); form a *baseline daily EW* of top-N names.
3) Create variants with:
   - Rebalance cadence: DAILY / WEEKLY(Fri-like last trading day) / MONTHLY(month-end)
   - Hysteresis bands (bps): 0 / 10 / 25 / 50  (only apply weight changes > band)
4) Compute daily portfolio returns and day-to-day turnover: tau_t = 0.5 * sum_i |w_t - w_{t-1}|
5) Annualize turnover and estimate cost drag = turnover * COST_BPS_PER_TURN
6) Choose the lowest cost_drag with tracking_corr >= 0.95 vs baseline.

Knobs to tune
-------------
- N_TOP_NAMES: number of names in the portfolio (default 20 or min(universe,20))
- LOOKBACK: signal lookback (63 trading days)
- COST_BPS_PER_TURN: effective round-trip cost per 100% turnover (default 35 bps)
- MIN_TRACKING_CORR: minimal correlation vs baseline to accept recommendation

This is *paper-only* optimization to cut churn before execution upgrades (W40).
"""

ROOT = Path(r"F:\Projects\trading_stack_py")
REPORTS = ROOT / "reports"
DATA = ROOT / "data" / "prices"

W11_CSV = REPORTS / "wk11_blend_targets.csv"
TURNOVER_CSV = REPORTS / "wk38_turnover_profile.csv"
RECO_JSON = REPORTS / "wk38_turnover_reco.json"

# --- knobs ---
LOOKBACK = 63
N_TOP_NAMES_DEFAULT = 20
COST_BPS_PER_TURN = 35.0  # conservative; your budget target was <=45 bps all-in
MIN_TRACKING_CORR = 0.95
BANDS_BPS = [0.0, 10.0, 25.0, 50.0]
CADENCES = ["DAILY", "WEEKLY", "MONTHLY"]


# ---------- helpers ----------
def _read_universe() -> list[str]:
    if not W11_CSV.exists():
        raise FileNotFoundError(f"Missing {W11_CSV} (run W11 first).")
    df = pd.read_csv(W11_CSV)
    cols = {c.lower(): c for c in df.columns}
    if "ticker" not in cols:
        raise ValueError("wk11_blend_targets.csv must contain 'ticker'")
    tickers = sorted(pd.Series(df[cols["ticker"]]).astype(str).unique())
    return tickers


def _read_prices(ticker: str) -> pd.DataFrame | None:
    p = DATA / f"{ticker}.parquet"
    if not p.exists():
        return None
    try:
        d = pd.read_parquet(p)
    except Exception:
        return None
    cols = {c.lower(): c for c in d.columns}
    dcol = cols.get("date") or cols.get("dt")
    ccol = cols.get("close") or cols.get("px_close") or cols.get("price")
    if not dcol or not ccol:
        return None
    out = d[[dcol, ccol]].rename(columns={dcol: "date", ccol: "close"}).copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna().sort_values("date")
    return out


def _panel(tickers: list[str]) -> pd.DataFrame:
    frames = []
    for t in tickers:
        px = _read_prices(t)
        if px is None or px.empty:
            continue
        frames.append(px.set_index("date").rename(columns={"close": t})[[t]])
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1, join="outer").sort_index()


def _rebalance_mask(idx: pd.DatetimeIndex, cadence: str) -> pd.Series:
    """Return a boolean mask (index=idx) marking rebalance days for the cadence.
    This version avoids .groupby(...).tail(1) to stay compatible with older pandas.
    """
    if cadence == "DAILY":
        return pd.Series(True, index=idx)

    if cadence == "WEEKLY":
        per = pd.Series(idx.to_period("W"), index=idx)
        mask = per.ne(per.shift(-1))  # True on last trading day of each week
        if len(mask) > 0:
            mask.iloc[-1] = True
        return mask

    if cadence == "MONTHLY":
        per = pd.Series(idx.to_period("M"), index=idx)
        mask = per.ne(per.shift(-1))  # True on last trading day of each month
        if len(mask) > 0:
            mask.iloc[-1] = True
        return mask

    raise ValueError(cadence)


def _topn_weights(scores_row: pd.Series, n: int) -> pd.Series:
    s = scores_row.dropna()
    if s.empty:
        return pd.Series(dtype=float)
    keep = s.nlargest(n).index
    w = pd.Series(0.0, index=s.index)
    if len(keep) > 0:
        w.loc[keep] = 1.0 / len(keep)
    return w


def _apply_hysteresis(
    prev_w: pd.Series, target_w: pd.Series, band_bps: float
) -> pd.Series:
    """
    Only move weight where |delta| > band. band_bps in basis points (1 bps = 0.0001).
    """
    if prev_w is None or prev_w.empty:
        return target_w.copy()
    band = band_bps / 10000.0
    delta = (target_w - prev_w).fillna(0.0)
    adj = prev_w.copy()
    big = delta.abs() > band
    adj.loc[big] = prev_w.loc[big] + delta.loc[big]
    # renormalize to 1 if any drift remains
    s = adj.clip(lower=0).sum()
    if s > 0:
        adj = adj.clip(lower=0) / s
    return adj


def _portfolio_returns(weights: pd.DataFrame, rets: pd.DataFrame) -> pd.Series:
    W = weights.reindex_like(rets).fillna(0.0)
    return (W * rets).sum(axis=1)


def _turnover_series(W: pd.DataFrame) -> pd.Series:
    # tau_t = 0.5 * sum |w_t - w_{t-1}|
    dW = (W - W.shift(1)).abs()
    return 0.5 * dW.sum(axis=1)


@dataclass
class ResultRow:
    setting: str
    rebalance: str
    band_bps: float
    annual_turnover: float
    cost_drag_bps: float
    tracking_corr: float
    tracking_rmse_bps: float
    notes: str


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    tickers = _read_universe()
    if len(tickers) == 0:
        pd.DataFrame(
            columns=[
                "setting",
                "rebalance",
                "band_bps",
                "annual_turnover",
                "cost_drag_bps",
                "tracking_corr",
                "tracking_rmse_bps",
                "notes",
            ]
        ).to_csv(TURNOVER_CSV, index=False)
        RECO_JSON.write_text(
            json.dumps({"recommended": None, "tested": 0}, indent=2), encoding="utf-8"
        )
        print(json.dumps({"rows": 0, "reason": "empty universe"}, indent=2))
        return

    panel = _panel(tickers)
    if panel.empty:
        pd.DataFrame(
            columns=[
                "setting",
                "rebalance",
                "band_bps",
                "annual_turnover",
                "cost_drag_bps",
                "tracking_corr",
                "tracking_rmse_bps",
                "notes",
            ]
        ).to_csv(TURNOVER_CSV, index=False)
        RECO_JSON.write_text(
            json.dumps({"recommended": None, "tested": 0}, indent=2), encoding="utf-8"
        )
        print(json.dumps({"rows": 0, "reason": "no price panel"}, indent=2))
        return

    # working window: drop initial NaNs from lookback
    rets = panel.pct_change()
    scores = panel / panel.shift(LOOKBACK) - 1.0
    dates = panel.index
    n_top = min(N_TOP_NAMES_DEFAULT, panel.shape[1]) if panel.shape[1] > 0 else 0
    if n_top == 0:
        raise RuntimeError("No tickers with price data.")

    # Baseline: DAILY, no hysteresis, top-N EW
    mask_daily = _rebalance_mask(dates, "DAILY")
    W_base = []
    prev = None
    for dt in dates:
        if not mask_daily.loc[dt]:
            W_base.append(
                prev if prev is not None else pd.Series(0.0, index=panel.columns)
            )
            continue
        row = scores.loc[dt]
        w_t = _topn_weights(row, n_top)
        prev = w_t.reindex(panel.columns).fillna(0.0)
        W_base.append(prev)
    W_base = pd.DataFrame(W_base, index=dates, columns=panel.columns).fillna(0.0)
    R_base = _portfolio_returns(W_base, rets).fillna(0.0)

    results: list[ResultRow] = []
    tested = 0

    for cadence in CADENCES:
        mask = _rebalance_mask(dates, cadence)
        for band in BANDS_BPS:
            tested += 1
            prev = None
            rows = []
            for dt in dates:
                if mask.loc[dt]:
                    tgt = (
                        _topn_weights(scores.loc[dt], n_top)
                        .reindex(panel.columns)
                        .fillna(0.0)
                    )
                    w_t = _apply_hysteresis(prev, tgt, band)
                else:
                    w_t = (
                        prev
                        if prev is not None
                        else pd.Series(0.0, index=panel.columns)
                    )
                prev = w_t
                rows.append(w_t)
            W = pd.DataFrame(rows, index=dates, columns=panel.columns).fillna(0.0)

            # Drop warmup
            W = W.iloc[LOOKBACK + 1 :].copy()
            R = _portfolio_returns(W, rets).iloc[LOOKBACK + 1 :].fillna(0.0)
            Rb = R_base.loc[R.index].fillna(0.0)

            # stats
            tau = _turnover_series(W).loc[R.index].fillna(0.0)
            annual_turn = float(tau.mean() * 252.0)  # ~252 trading days
            cost_drag = float(
                annual_turn * (COST_BPS_PER_TURN / 10000.0) * 1e4
            )  # in bps
            # tracking vs baseline
            corr = (
                float(pd.Series(R).corr(pd.Series(Rb)))
                if R.std(ddof=1) > 0 and Rb.std(ddof=1) > 0
                else np.nan
            )
            rmse = float(np.sqrt(((R - Rb) ** 2).mean()) * 1e4)  # daily RMSE in bps

            results.append(
                ResultRow(
                    setting=f"{cadence}_band{int(band)}",
                    rebalance=cadence,
                    band_bps=band,
                    annual_turnover=round(annual_turn, 2),
                    cost_drag_bps=round(cost_drag, 2),
                    tracking_corr=round(corr, 4) if np.isfinite(corr) else np.nan,
                    tracking_rmse_bps=round(rmse, 2),
                    notes=f"N={n_top}, lookback={LOOKBACK}, window_days={len(R)}",
                )
            )

    out = pd.DataFrame([r.__dict__ for r in results])
    out = out.sort_values(["cost_drag_bps", "band_bps", "rebalance"])
    out.to_csv(TURNOVER_CSV, index=False)

    # recommendation: min cost subject to corr >= MIN_TRACKING_CORR
    cand = out[out["tracking_corr"] >= MIN_TRACKING_CORR].copy()
    if cand.empty:
        # fallback: best corr, then min cost
        cand = out.sort_values(
            ["tracking_corr", "cost_drag_bps"], ascending=[False, True]
        ).head(1)
    reco = cand.sort_values("cost_drag_bps", ascending=True).iloc[0].to_dict()

    RECO_JSON.write_text(
        json.dumps(
            {
                "recommended": reco,
                "tested": tested,
                "window_days": (
                    int(reco["notes"].split("window_days=")[-1])
                    if "window_days=" in reco["notes"]
                    else None
                ),
                "knobs": {
                    "lookback": LOOKBACK,
                    "top_names": n_top,
                    "cost_bps_per_turn": COST_BPS_PER_TURN,
                    "min_tracking_corr": MIN_TRACKING_CORR,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "turnover_csv": str(TURNOVER_CSV),
                "reco_json": str(RECO_JSON),
                "recommended": reco,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

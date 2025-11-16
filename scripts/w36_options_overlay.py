from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

"""
W36 — Options Overlay (Protective Puts / Covered Calls)

Inputs
- reports/wk11_blend_targets.csv  (date,ticker,target_w,...)
- data/prices/*.parquet           (per-ticker EOD with columns including date, close)
  (You already seeded missing ones earlier for W4; we'll use whatever is present.)

What we do
1) Build a daily portfolio NAV (rebalance to target_w each date; close-to-close returns).
2) Estimate realized vol (rolling 30d) as a proxy for implied vol (with a min floor).
3) Grid-test overlays:
   - Protective put: hedge_ratio in {0.3, 0.5, 0.7, 1.0}, tenor=30d, moneyness in {ATM, 95%}.
   - Covered call: call_coverage in {0.3, 0.5, 0.7, 1.0}, tenor=30d, moneyness in {ATM, 105%}.
   We cost them using Black-Scholes (r=0), IV = max(realized_30d, MIN_IV).

4) Evaluate:
   - premium_cost_bps (per 30d, annualized),
   - median and P90 daily slippage of overlay ≈ 0 (execution cost ignored here),
   - drawdown metrics on the overlayed NAV,
   - shock protection at −10% and −20% one-day shocks.

Outputs
- reports/wk36_options_overlay.csv     (grid summary with key metrics)
- reports/w36_overlay_detail.csv       (top candidates expanded)
- reports/w36_overlay_summary.json     (chosen recommendation + knobs)
"""

ROOT = Path(r"F:\Projects\trading_stack_py")
REPORTS = ROOT / "reports"
DATA = ROOT / "data" / "prices"

TARGETS_CSV = REPORTS / "wk11_blend_targets.csv"
OUT_GRID_CSV = REPORTS / "wk36_options_overlay.csv"
OUT_DETAIL = REPORTS / "w36_overlay_detail.csv"
OUT_SUMMARY = REPORTS / "w36_overlay_summary.json"

# --- Knobs ---
RFR = 0.00  # risk-free for BS (daily compounding not used; T in yrs)
DAYS_IN_YEAR = 252
TENOR_DAYS = 30
MIN_IV_ANN = 0.18  # if realized vol lower, floor IV here
MAX_IV_ANN = 0.60  # clamp noisy IV
IV_SMOOTH_DAYS = 30

# Protective puts grid
PUT_HEDGE_RATIOS = [0.3, 0.5, 0.7, 1.0]
PUT_MONEYNESS = [1.00, 0.95]  # K = m * spot (ATM, 95% strike)
# Covered calls grid
CALL_COVER_RATIOS = [0.3, 0.5, 0.7, 1.0]
CALL_MONEYNESS = [1.00, 1.05]  # K = m * spot (ATM, 105%)

# Scoring weights (lower is better for cost; higher is better for protection)
ALPHA_COST = 1.0
ALPHA_DD_REDUCT = 1.5
ALPHA_SHOCK10 = 0.5
ALPHA_SHOCK20 = 1.0


# ----------------- Utilities -----------------
def _ann_to_sday(sig_ann: float) -> float:
    return max(1e-8, sig_ann / math.sqrt(DAYS_IN_YEAR))


def _sday_to_ann(sig_day: float) -> float:
    return max(1e-8, sig_day * math.sqrt(DAYS_IN_YEAR))


def _black_scholes_premium(
    spot: float, strike: float, iv_ann: float, days: int, is_call: bool
) -> float:
    """
    Black-Scholes option price (no dividends, r=0). Returns premium in same units as spot (per unit).
    """
    if spot <= 0 or strike <= 0 or iv_ann <= 0 or days <= 0:
        return 0.0
    T = max(1e-8, days / DAYS_IN_YEAR)
    sigma = max(1e-8, iv_ann)
    d1 = (math.log(spot / strike) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    from math import erf

    def N(x: float) -> float:
        return 0.5 * (1.0 + erf(x / math.sqrt(2.0)))

    if is_call:
        return spot * N(d1) - strike * N(d2)
    else:
        # put-call parity with r=0: P = K*N(-d2) - S*N(-d1)
        return strike * N(-d2) - spot * N(-d1)


def _read_prices_for(ticker: str) -> pd.DataFrame | None:
    p = DATA / f"{ticker}.parquet"
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p)
    except Exception:
        return None
    cols = {c.lower(): c for c in df.columns}
    dcol = cols.get("date") or cols.get("dt")
    ccol = cols.get("close") or cols.get("px_close") or cols.get("price")
    if not dcol or not ccol:
        return None
    out = df[[dcol, ccol]].copy().rename(columns={dcol: "date", ccol: "close"})
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date", "close"]).sort_values("date")
    return out


def _load_panel(targets: pd.DataFrame) -> pd.DataFrame:
    # Merge all available tickers' close prices into a wide panel
    tickers = sorted(targets["ticker"].unique())
    frames = []
    for t in tickers:
        px = _read_prices_for(t)
        if px is None or px.empty:
            continue
        px = px.rename(columns={"close": t})
        frames.append(px.set_index("date")[[t]])
    if not frames:
        return pd.DataFrame()
    panel = pd.concat(frames, axis=1, join="outer").sort_index()
    return panel


def _build_nav(targets: pd.DataFrame, prices_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Rebalance to target weights each date (close-to-close). Returns df with date, ret, nav.
    """
    if prices_wide.empty:
        return pd.DataFrame(columns=["date", "ret", "nav"])

    # Compute returns per ticker
    rets = prices_wide.pct_change().replace([np.inf, -np.inf], np.nan)

    # Ensure we only use dates present in both targets and prices
    td = targets.copy()
    td["date"] = pd.to_datetime(td["date"])
    # Normalize weights each date to sum=1 over available tickers that have price that day
    w = (
        td.pivot(index="date", columns="ticker", values="target_w")
        .sort_index()
        .fillna(0.0)
    )
    # Optional: normalize row-wise to 1.0 to avoid scale issues if sums drift
    ws = w.div(w.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    # Align to returns index
    wsp = ws.reindex(rets.index).fillna(method="ffill").fillna(0.0)
    port_ret = (wsp * rets).sum(axis=1)
    nav = (1.0 + port_ret.fillna(0.0)).cumprod()
    out = pd.DataFrame({"date": rets.index, "ret": port_ret.values, "nav": nav.values})
    out = out.dropna(subset=["date"])
    return out


def _drawdown_stats(nav: pd.Series) -> dict:
    if nav.isna().all() or nav.empty:
        return {"max_dd": 0.0}
    roll_max = nav.cummax()
    dd = nav / roll_max - 1.0
    return {"max_dd": float(dd.min())}


def _estimate_iv(nav_ret: pd.Series) -> pd.Series:
    # Realized vol (rolling 30d) → annualize → clamp → smooth
    day_vol = nav_ret.rolling(30).std().clip(lower=1e-8)
    iv_ann = _sday_to_ann(day_vol).clip(lower=MIN_IV_ANN, upper=MAX_IV_ANN)
    iv_ann = iv_ann.rolling(IV_SMOOTH_DAYS, min_periods=1).mean()
    return iv_ann


def _apply_protective_puts(
    nav_df: pd.DataFrame, hedge_ratio: float, moneyness: float, iv_ann: pd.Series
) -> pd.DataFrame:
    """
    Roll 30d ATM/OTM puts each day (conceptually). Premium accrued daily as 1/TENOR of 30d price.
    Apply payoff only on down days (approx with daily delta ~ convexity ignored).
    We approximate by treating each day’s overlay on spot=nav level.
    """
    df = nav_df.copy()
    df["spot"] = df["nav"]
    df["iv"] = iv_ann.reindex(df["date"]).fillna(method="ffill").fillna(MIN_IV_ANN)
    # daily premium per unit notional hedged
    prem_30d = _black_scholes_premium(
        1.0,
        moneyness * 1.0,
        df["iv"].iloc[0] if not df["iv"].empty else MIN_IV_ANN,
        TENOR_DAYS,
        is_call=False,
    )
    # Allow iv to vary by day → recompute premium day-wise
    prem_day = []
    put_pay = []
    for i, r in df.iterrows():
        iv = float(df.at[i, "iv"])
        s = float(df.at[i, "spot"])
        if not math.isfinite(s) or s <= 0:
            prem_day.append(0.0)
            put_pay.append(0.0)
            continue
        k = moneyness * s
        prem = _black_scholes_premium(s, k, iv, TENOR_DAYS, is_call=False) / TENOR_DAYS
        prem_day.append(prem)
        # one-day payoff approximation when next-day drop moves below strike
        # Approximated by max(K - S_next, 0). We proxy with daily return only (ignoring path within day).
        put_pay.append(
            0.0
        )  # payoff applied at next step via shock tests more reliably than daily; keep 0 here.

    df["prem_day"] = prem_day
    # Hedge reduces effective return on down days (delta effect). Simpler: apply linear down-day cushion:
    # Cushion ≈ hedge_ratio * max(0, -ret)
    df["ret_overlay"] = df["ret"] + hedge_ratio * df["ret"].clip(
        upper=0.0
    )  # reduce losses proportionally
    df["nav_overlay_gross"] = (1.0 + df["ret_overlay"].fillna(0.0)).cumprod()
    # subtract premium drag (as bps of notional hedged on the NAV base)
    # Daily drag ≈ hedge_ratio * prem_day / spot
    df["drag"] = hedge_ratio * df["prem_day"] / df["spot"].replace(0, np.nan)
    df["nav_overlay"] = (
        df["nav_overlay_gross"] * (1.0 - df["drag"].fillna(0.0))
    ).cumprod()
    return df[["date", "ret", "nav", "nav_overlay", "drag"]]


def _apply_covered_calls(
    nav_df: pd.DataFrame, call_coverage: float, moneyness: float, iv_ann: pd.Series
) -> pd.DataFrame:
    """
    Sell 30d calls against a fraction of holdings. Premium adds to returns; upside capped beyond strike.
    Approximation: on up days, cap gain by (call_coverage * excess above strike).
    """
    df = nav_df.copy()
    df["spot"] = df["nav"]
    df["iv"] = iv_ann.reindex(df["date"]).fillna(method="ffill").fillna(MIN_IV_ANN)
    prem_day = []
    for i, r in df.iterrows():
        iv = float(df.at[i, "iv"])
        s = float(df.at[i, "spot"])
        if not math.isfinite(s) or s <= 0:
            prem_day.append(0.0)
            continue
        k = moneyness * s
        prem = _black_scholes_premium(s, k, iv, TENOR_DAYS, is_call=True) / TENOR_DAYS
        prem_day.append(prem)
    df["prem_day"] = prem_day
    # Upside cap approximation: reduce positive returns proportionally to coverage when big up moves
    cap_factor = moneyness - 1.0  # room before cap kicks; 0 for ATM
    # If cap_factor <= 0 (ATM), any positive return is partially capped. If 0.05 (105%), only returns >5% are capped.
    up = df["ret"].clip(lower=0.0)
    capped_extra = (up - cap_factor).clip(lower=0.0)
    cap_penalty = call_coverage * capped_extra
    ret_after_cap = df["ret"] - cap_penalty
    # Add premium (as % of NAV) on covered fraction
    prem_pct = call_coverage * (df["prem_day"] / df["spot"].replace(0, np.nan)).fillna(
        0.0
    )
    df["ret_overlay"] = ret_after_cap + prem_pct
    df["nav_overlay"] = (1.0 + df["ret_overlay"].fillna(0.0)).cumprod()
    return df[["date", "ret", "nav", "nav_overlay"]]


def _shock_loss(
    nav_val: float,
    shock_pct: float,
    overlay_type: str,
    cover: float,
    moneyness: float,
    iv_ann: float,
) -> float:
    """
    Estimate 1-day loss under a shock with overlay.
    For puts: payoff ~ hedge_ratio * max(K - S*(1+shock), 0)
    For calls: upside cap irrelevant on a down shock; premium helps a little.
    Returns loss (%) relative to original NAV.
    """
    s = max(1e-8, nav_val)
    s_next = s * (1.0 + shock_pct)
    if overlay_type == "PUT":
        k = moneyness * s
        payoff = cover * max(k - s_next, 0.0)
        # premium for that day (approx)
        prem = cover * (
            _black_scholes_premium(s, k, iv_ann, TENOR_DAYS, is_call=False) / TENOR_DAYS
        )
        nav_after = s_next + payoff - prem
        return (nav_after / s) - 1.0
    else:
        # CALL: on down shock, no cap; small premium gain
        k = moneyness * s
        prem = cover * (
            _black_scholes_premium(s, k, iv_ann, TENOR_DAYS, is_call=True) / TENOR_DAYS
        )
        nav_after = s_next + prem
        return (nav_after / s) - 1.0


def _score_row(
    kind: str,
    cost_bps_ann: float,
    base_dd: float,
    dd_ov: float,
    shock10: float,
    shock20: float,
) -> float:
    # Max drawdown numbers are negative; drawdown reduction is (dd_ov - base_dd) (less negative is better)
    dd_reduct = dd_ov - base_dd  # e.g., base -0.30, overlay -0.22 → +0.08
    # Shock improvements vs base (-0.10 becomes closer to 0 is better)
    # Here we assume base shock = shock_pct; add positive improvement magnitude
    # We’ll use absolute protection improvement toward 0: higher is better
    prot10 = 0.0 - (-0.10 - shock10)  # if overlay shock10 = -0.07, improvement = 0.03
    prot20 = 0.0 - (-0.20 - shock20)
    # Lower score better; cost penalizes linearly, protections rewarded
    return (
        ALPHA_COST * (cost_bps_ann / 10000.0)
        - ALPHA_DD_REDUCT * dd_reduct
        - ALPHA_SHOCK10 * prot10
        - ALPHA_SHOCK20 * prot20
    )


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    if not TARGETS_CSV.exists():
        raise FileNotFoundError(f"Missing {TARGETS_CSV}; run W11 first.")

    tdf = pd.read_csv(TARGETS_CSV)
    cols = {c.lower(): c for c in tdf.columns}
    if "date" not in cols or "ticker" not in cols or "target_w" not in cols:
        raise ValueError("wk11_blend_targets.csv must contain date,ticker,target_w")
    tdf = tdf.rename(
        columns={
            cols["date"]: "date",
            cols["ticker"]: "ticker",
            cols["target_w"]: "target_w",
        }
    )
    tdf["date"] = pd.to_datetime(tdf["date"], errors="coerce")
    tdf = tdf.dropna(subset=["date"])
    tdf["ticker"] = tdf["ticker"].astype(str)
    tdf["target_w"] = pd.to_numeric(tdf["target_w"], errors="coerce").fillna(0.0)

    prices_wide = _load_panel(tdf)
    nav_df = _build_nav(tdf, prices_wide)
    if nav_df.empty:
        # fallback: flat NAV if prices missing
        nav_df = pd.DataFrame(
            {
                "date": pd.to_datetime(sorted(tdf["date"].unique())),
                "ret": 0.0,
                "nav": 1.0,
            }
        )
    base_dd = _drawdown_stats(nav_df["nav"])["max_dd"]

    # IV proxy from portfolio returns
    iv_ann_series = _estimate_iv(nav_df["ret"])

    rows = []

    # Protective puts
    if not nav_df.empty:
        for hr in PUT_HEDGE_RATIOS:
            for m in PUT_MONEYNESS:
                put_df = _apply_protective_puts(
                    nav_df, hedge_ratio=hr, moneyness=m, iv_ann=iv_ann_series
                )
                dd_put = _drawdown_stats(put_df["nav_overlay"])["max_dd"]
                # premium drag estimate (annualized) = mean(daily drag)*252
                cost_bps_ann = 1e4 * put_df["drag"].fillna(0.0).mean() * DAYS_IN_YEAR
                # shocks from last valid day
                last_idx = put_df["nav"].last_valid_index()
                last_nav = (
                    float(put_df.loc[last_idx, "nav"]) if last_idx is not None else 1.0
                )
                iv_last = (
                    float(iv_ann_series.iloc[-1])
                    if not iv_ann_series.empty
                    else MIN_IV_ANN
                )
                s10 = _shock_loss(last_nav, -0.10, "PUT", hr, m, iv_last)  # return (%)
                s20 = _shock_loss(last_nav, -0.20, "PUT", hr, m, iv_last)
                score = _score_row("PUT", cost_bps_ann, base_dd, dd_put, s10, s20)

                rows.append(
                    {
                        "kind": "PUT",
                        "hedge_ratio": hr,
                        "moneyness": m,
                        "tenor_days": TENOR_DAYS,
                        "cost_bps_ann": round(float(cost_bps_ann), 2),
                        "base_max_dd": round(float(base_dd), 4),
                        "overlay_max_dd": round(float(dd_put), 4),
                        "dd_reduction": round(float(dd_put - base_dd), 4),
                        "shock10_ret": round(float(s10), 4),
                        "shock20_ret": round(float(s20), 4),
                        "score": round(float(score), 6),
                    }
                )

    # Covered calls
    if not nav_df.empty:
        for cr in CALL_COVER_RATIOS:
            for m in CALL_MONEYNESS:
                call_df = _apply_covered_calls(
                    nav_df, call_coverage=cr, moneyness=m, iv_ann=iv_ann_series
                )
                dd_call = _drawdown_stats(call_df["nav_overlay"])["max_dd"]
                # premium adds return; we approximate cost as negative bps (i.e., benefit)
                # but for ranking consistency, we’ll keep cost_bps_ann ~ 0 here since premium already boosts NAV.
                cost_bps_ann = 0.0
                last_idx = call_df["nav"].last_valid_index()
                last_nav = (
                    float(call_df.loc[last_idx, "nav"]) if last_idx is not None else 1.0
                )
                iv_last = (
                    float(iv_ann_series.iloc[-1])
                    if not iv_ann_series.empty
                    else MIN_IV_ANN
                )
                s10 = _shock_loss(last_nav, -0.10, "CALL", cr, m, iv_last)
                s20 = _shock_loss(last_nav, -0.20, "CALL", cr, m, iv_last)
                score = _score_row("CALL", cost_bps_ann, base_dd, dd_call, s10, s20)

                rows.append(
                    {
                        "kind": "CALL",
                        "call_coverage": cr,
                        "moneyness": m,
                        "tenor_days": TENOR_DAYS,
                        "cost_bps_ann": round(float(cost_bps_ann), 2),
                        "base_max_dd": round(float(base_dd), 4),
                        "overlay_max_dd": round(float(dd_call), 4),
                        "dd_reduction": round(float(dd_call - base_dd), 4),
                        "shock10_ret": round(float(s10), 4),
                        "shock20_ret": round(float(s20), 4),
                        "score": round(float(score), 6),
                    }
                )

    grid = pd.DataFrame(rows)
    if not grid.empty:
        # Lower score is better
        best_idx = grid["score"].idxmin()
        grid["recommend"] = False
        grid.loc[best_idx, "recommend"] = True
        best = grid.loc[best_idx].to_dict()
    else:
        best = None
        grid = pd.DataFrame(
            columns=[
                "kind",
                "hedge_ratio",
                "call_coverage",
                "moneyness",
                "tenor_days",
                "cost_bps_ann",
                "base_max_dd",
                "overlay_max_dd",
                "dd_reduction",
                "shock10_ret",
                "shock20_ret",
                "score",
                "recommend",
            ]
        )

    # Save
    REPORTS.mkdir(parents=True, exist_ok=True)
    grid.to_csv(OUT_GRID_CSV, index=False)
    top = grid.nsmallest(12, "score") if not grid.empty else grid
    top.to_csv(OUT_DETAIL, index=False)

    summary = {
        "rows": int(grid.shape[0]),
        "best": best,
        "notes": "Protective puts add drag (premium) but reduce drawdowns/shocks; covered calls add income but cap upside.",
        "files": {"grid_csv": str(OUT_GRID_CSV), "detail_csv": str(OUT_DETAIL)},
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"grid_csv": str(OUT_GRID_CSV), "best": best}, indent=2))


if __name__ == "__main__":
    main()

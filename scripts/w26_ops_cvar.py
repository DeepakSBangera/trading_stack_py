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

# Inputs (soft dependencies; code falls back gracefully)
TARGETS_CSV = REPORTS / "wk11_blend_targets.csv"  # weights
VOL_DIAG_JSON = REPORTS / "w4_diag_summary.json"  # has "overall_vol_ann_est"
# Outputs
RET_SERIES_CSV = REPORTS / "w26_daily_portfolio_returns.csv"
STRESS_TABLE_CSV = REPORTS / "w26_stress_table.csv"
SUMMARY_CSV = REPORTS / "wk26_ops_cvar.csv"

# Settings
DEFAULT_PORTFOLIO_NOTIONAL_INR = 10_000_000
TRADING_DAYS_LOOKBACK = 360
CONF_LEVELS = [0.95, 0.99]  # for VaR & CVaR

PRICE_DATE_CANDS = ["date", "dt"]
PRICE_CLOSE_CANDS = ["close", "px_close", "price", "adj_close"]


def _pick(df: pd.DataFrame, cands: list[str]) -> str | None:
    cols = {c.lower(): c for c in df.columns}
    for k in cands:
        if k in cols:
            return cols[k]
    for c in df.columns:
        lc = c.lower().replace(" ", "").replace("-", "_")
        for k in cands:
            if lc == k.replace(" ", "").replace("-", "_"):
                return c
    return None


def _load_weights() -> tuple[pd.DataFrame, dt.date]:
    if not TARGETS_CSV.exists():
        raise FileNotFoundError(f"Missing {TARGETS_CSV}. Run W11 first.")
    t = pd.read_csv(TARGETS_CSV)
    dcol = _pick(t, ["date", "trading_day", "dt"])
    wcol = _pick(t, ["target_w", "weight", "w", "base_w"])
    tic = _pick(t, ["ticker", "symbol", "name"])
    if not (dcol and wcol and tic):
        raise ValueError("wk11_blend_targets.csv missing one of: date/ticker/target_w.")
    t[dcol] = pd.to_datetime(t[dcol], errors="coerce").dt.date
    last_day = t[dcol].max()
    snap = t[t[dcol] == last_day][[tic, wcol]].copy()
    snap.columns = ["ticker", "w"]
    # normalize weights just in case
    snap["w"] = pd.to_numeric(snap["w"], errors="coerce").fillna(0.0)
    tot = snap["w"].abs().sum()
    if tot > 0:
        snap["w"] = snap["w"] / tot
    return snap, last_day


def _load_px_series(ticker: str) -> pd.DataFrame | None:
    p = DATA / f"{ticker}.parquet"
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p)
        dcol = _pick(df, PRICE_DATE_CANDS)
        ccol = _pick(df, PRICE_CLOSE_CANDS)
        if not (dcol and ccol):
            return None
        df = df[[dcol, ccol]].copy()
        df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
        df = df.dropna().sort_values(dcol)
        df.columns = ["date", "close"]
        return df
    except Exception:
        return None


def _portfolio_returns(
    weights: pd.DataFrame, last_day: dt.date
) -> tuple[pd.Series, bool]:
    """
    Build daily portfolio returns over the LOOKBACK window using px parquet if available.
    Fallback: synthetic Gaussian using annual vol from W4 diag.
    Returns: (pd.Series daily_ret indexed by date, fallback_used)
    """
    # Try historical
    rets = []
    names = []
    for tck in weights["ticker"]:
        px = _load_px_series(str(tck))
        if px is None or px.empty:
            continue
        px = px[px["date"] <= pd.Timestamp(last_day)]
        px = px.tail(TRADING_DAYS_LOOKBACK + 1)
        if len(px) < 30:  # too short
            continue
        r = px["close"].pct_change().dropna()
        r.index = px["date"].iloc[1:].values
        rets.append(r)
        names.append(tck)
    if rets:
        # Align by date, fill missing with 0 (conservative)
        mat = pd.concat(rets, axis=1)
        mat.columns = names
        w = weights.set_index("ticker")["w"].reindex(mat.columns).fillna(0.0)
        port = (mat * w.values).sum(axis=1)
        port.name = "ret"
        return port.sort_index(), False

    # Fallback synthetic
    # Get ann vol estimate from W4; else default 20% ann
    vol_ann = 0.20
    try:
        if VOL_DIAG_JSON.exists():
            jd = json.loads(Path(VOL_DIAG_JSON).read_text(encoding="utf-8"))
            v = jd.get("overall_vol_ann_est", None)
            if isinstance(v, (int, float)) and v > 0:
                vol_ann = float(v)
    except Exception:
        pass
    vol_daily = vol_ann / math.sqrt(252.0)
    np.random.seed(42)
    synth = np.random.normal(loc=0.0, scale=vol_daily, size=TRADING_DAYS_LOOKBACK)
    idx = pd.date_range(
        end=pd.Timestamp(last_day), periods=TRADING_DAYS_LOOKBACK, freq="B"
    )
    ser = pd.Series(synth, index=idx, name="ret")
    return ser, True


def _var_cvar(series: pd.Series, alpha: float) -> tuple[float, float]:
    """Historical (empirical) VaR/CVaR on returns (loss is negative). Returns positive numbers as loss %."""
    s = series.dropna().astype(float).values
    if len(s) == 0:
        return 0.0, 0.0
    # VaR_alpha: the (1-alpha) lower quantile of returns
    q = np.quantile(s, 1 - alpha)
    var = max(0.0, -q) * 100.0
    # CVaR (Expected Shortfall): mean of tail below the VaR threshold
    tail = s[s <= q]
    cvar = max(0.0, -tail.mean()) * 100.0 if len(tail) else 0.0
    return float(var), float(cvar)


def _stress_table(notional_inr: float) -> pd.DataFrame:
    # Simple, interpretable one-day shocks
    scenarios = [
        ("Daily -5%", -5.0),
        ("Daily -10%", -10.0),
        ("Daily -15%", -15.0),
        ("Crisis -20%", -20.0),
        ("Crisis -30%", -30.0),
        ("Flash -9.4% (2008-like)", -9.4),
        ("Flash -13% (2020-like)", -13.0),
    ]
    rows = []
    for name, loss_pct in scenarios:
        rows.append(
            {
                "scenario": name,
                "loss_pct": loss_pct,
                "est_loss_inr": round(abs(loss_pct) / 100.0 * notional_inr, 2),
            }
        )
    return pd.DataFrame(rows)


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)

    weights, last_day = _load_weights()
    port_ret, fallback = _portfolio_returns(weights, last_day)

    # Save series used
    out_ser = pd.DataFrame({"date": port_ret.index, "ret": port_ret.values})
    out_ser.to_csv(RET_SERIES_CSV, index=False)

    # VaR/CVaR for both 95% and 99%
    metrics = {}
    for cl in CONF_LEVELS:
        var, cvar = _var_cvar(port_ret, cl)
        metrics[f"VaR_{int(cl * 100)}_pct"] = round(var, 3)
        metrics[f"CVaR_{int(cl * 100)}_pct"] = round(cvar, 3)

    # Stress table
    stress = _stress_table(DEFAULT_PORTFOLIO_NOTIONAL_INR)
    stress.to_csv(STRESS_TABLE_CSV, index=False)

    # Summary
    summary = {
        "as_of": str(last_day),
        "days_used": int(port_ret.shape[0]),
        "fallback_synthetic": bool(fallback),
        **metrics,
        "ret_series_csv": str(RET_SERIES_CSV),
        "stress_table_csv": str(STRESS_TABLE_CSV),
    }
    pd.DataFrame([summary]).to_csv(SUMMARY_CSV, index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

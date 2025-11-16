from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

"""
W27 — Contextual Bandit (Rule Selection)
- Inputs (optional if present):
    reports/ic_timeseries.csv              # per-sleeve IC by date (cols: date, sleeve, ic)
    reports/wk17_ops_attrib.csv           # attribution by sleeve (cols: date, sleeve, ret or pnl)
    reports/canary_log.csv                # canary outcomes (cols: date, sleeve, realized_ret)
- Fallback (if none above exists):
    Synthetic 60B-day sleeve metrics (stable + noise), so pipeline always runs.
- Policy:
    Score = EWMA(mean_ret_bps) / (EWMA(vol_bps) + eps)  (simple risk-adjusted)
    Weight raw = softmax(score / TEMPERATURE)
    Apply safety:
        * Per-sleeve min/max caps
        * Max weekly delta from last policy (if exists)
    Normalize -> weights_final
- Outputs:
    reports/w27_bandit_selection.csv
    reports/w27_bandit_summary.json
    reports/w27_last_policy.csv   (for next week’s max-delta constraint)
"""

ROOT = Path(r"F:\Projects\trading_stack_py")
REPORTS = ROOT / "reports"

# Optional inputs
IC_CSV = REPORTS / "ic_timeseries.csv"
ATTRIB_CSV = REPORTS / "wk17_ops_attrib.csv"
CANARY_CSV = REPORTS / "canary_log.csv"

# Outputs
POLICY_CSV = REPORTS / "w27_bandit_selection.csv"
SUMMARY_JSON = REPORTS / "w27_bandit_summary.json"
LAST_POLICY = REPORTS / "w27_last_policy.csv"

# Sleeve universe (edit/extend later)
SLEEVES = ["momentum_core", "event_drift", "shortterm_rev"]

# Settings
LOOKBACK_DAYS = 60  # window for metric aggregation
EWMA_ALPHA_MEAN = 0.20  # smoothing for mean
EWMA_ALPHA_VOL = 0.20  # smoothing for vol
TEMPERATURE = 0.75  # softmax temperature (lower=peakier)
EPS = 1e-6

# Safety caps & constraints
MIN_W = 0.00  # min per sleeve
MAX_W = 0.70  # max per sleeve
MAX_DELTA = 0.15  # max weekly change per sleeve (absolute)
ROUND_TO = 0.0001  # round weights


def _now_ist_str() -> str:
    # simple local timestamp (host time); IST formatting not critical for logic
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _ensure_reports():
    REPORTS.mkdir(parents=True, exist_ok=True)


def _read_optional_csv(path: Path) -> pd.DataFrame | None:
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception:
        pass
    return None


def _ts_daterange_end_today(n=LOOKBACK_DAYS) -> pd.DatetimeIndex:
    return pd.date_range(end=pd.Timestamp.today().normalize(), periods=n, freq="B")


def _fallback_metrics() -> pd.DataFrame:
    """
    Build a synthetic per-sleeve daily metric frame:
    columns: date, sleeve, ret_bps (daily return in bps)
    """
    idx = _ts_daterange_end_today(LOOKBACK_DAYS)
    rng = np.random.default_rng(42)
    base = {
        "momentum_core": rng.normal(3.2, 12.0, len(idx)),  # mean 3.2 bps/d
        "event_drift": rng.normal(2.4, 10.0, len(idx)),
        "shortterm_rev": rng.normal(1.2, 15.0, len(idx)),
    }
    rows = []
    for sleeve, arr in base.items():
        for d, x in zip(idx, arr, strict=False):
            rows.append({"date": d, "sleeve": sleeve, "ret_bps": float(x)})
    return pd.DataFrame(rows)


def _load_metrics() -> pd.DataFrame:
    """
    Try to assemble per-sleeve daily ret_bps from any of the optional inputs.
    Priority:
      1) wk17_ops_attrib.csv with columns (date, sleeve, ret or pnl) -> convert to bps proxy
      2) ic_timeseries.csv with (date, sleeve, ic) -> map ic to ret_bps via scale
      3) canary_log.csv (date, sleeve, realized_ret) -> bps
      4) fallback synthetic
    """
    # 1) Attribution
    df = _read_optional_csv(ATTRIB_CSV)
    if df is not None and not df.empty:
        cols = {c.lower(): c for c in df.columns}
        dcol = cols.get("date")
        scol = cols.get("sleeve") or cols.get("bucket")
        rcol = cols.get("ret") or cols.get("return") or cols.get("pnl")
        if dcol and scol and rcol:
            out = df[[dcol, scol, rcol]].copy()
            out.columns = ["date", "sleeve", "val"]
            out["date"] = pd.to_datetime(out["date"], errors="coerce")
            # Assume val ~ daily return in decimal -> bps
            out["ret_bps"] = pd.to_numeric(out["val"], errors="coerce") * 1e4
            out = out.dropna(subset=["date", "sleeve", "ret_bps"])
            return out[["date", "sleeve", "ret_bps"]]

    # 2) IC time series (map IC to bps via scale)
    df = _read_optional_csv(IC_CSV)
    if df is not None and not df.empty:
        cols = {c.lower(): c for c in df.columns}
        dcol = cols.get("date")
        scol = cols.get("sleeve") or cols.get("signal") or cols.get("family")
        icol = cols.get("ic")
        if dcol and scol and icol:
            out = df[[dcol, scol, icol]].copy()
            out.columns = ["date", "sleeve", "ic"]
            out["date"] = pd.to_datetime(out["date"], errors="coerce")
            out["ret_bps"] = (
                pd.to_numeric(out["ic"], errors="coerce") * 12.0 * 1e1
            )  # heuristic scale
            out = out.dropna(subset=["date", "sleeve", "ret_bps"])
            return out[["date", "sleeve", "ret_bps"]]

    # 3) Canary realized returns
    df = _read_optional_csv(CANARY_CSV)
    if df is not None and not df.empty:
        cols = {c.lower(): c for c in df.columns}
        dcol = cols.get("date")
        scol = cols.get("sleeve")
        rcol = cols.get("realized_ret") or cols.get("ret")
        if dcol and scol and rcol:
            out = df[[dcol, scol, rcol]].copy()
            out.columns = ["date", "sleeve", "val"]
            out["date"] = pd.to_datetime(out["date"], errors="coerce")
            out["ret_bps"] = pd.to_numeric(out["val"], errors="coerce") * 1e4
            out = out.dropna(subset=["date", "sleeve", "ret_bps"])
            return out[["date", "sleeve", "ret_bps"]]

    # 4) fallback synthetic
    return _fallback_metrics()


def _clip_dates(df: pd.DataFrame) -> pd.DataFrame:
    # keep last LOOKBACK window
    if "date" not in df.columns:
        return df
    df = df.dropna(subset=["date"]).copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    mx = df["date"].max()
    if pd.isna(mx):
        return df
    mn = mx - pd.Timedelta(days=int(LOOKBACK_DAYS * 1.6))  # loose buffer
    return df[(df["date"] >= mn) & (df["date"] <= mx)].copy()


def _ewma(series: pd.Series, alpha: float) -> float:
    s = pd.Series(series).dropna().astype(float)
    if s.empty:
        return 0.0
    return float(s.ewm(alpha=alpha).mean().iloc[-1])


def _softmax(x: np.ndarray, temp: float) -> np.ndarray:
    x = np.array(x, dtype=float)
    x = (x - np.max(x)) / max(temp, EPS)
    ex = np.exp(x)
    z = ex.sum()
    return ex / z if z > 0 else np.ones_like(x) / len(x)


def _load_last_policy() -> pd.Series:
    if LAST_POLICY.exists():
        try:
            prev = pd.read_csv(LAST_POLICY)
            prev = prev.set_index("sleeve")["weight"]
            return prev
        except Exception:
            pass
    return pd.Series(dtype=float)


def _apply_safety_caps(weights: pd.Series, last: pd.Series) -> pd.Series:
    # per-sleeve min/max
    w = weights.clip(lower=MIN_W, upper=MAX_W)
    # max delta vs last policy
    if not last.empty:
        aligned_last = last.reindex(w.index).fillna(0.0)
        delta = (w - aligned_last).clip(lower=-MAX_DELTA, upper=MAX_DELTA)
        w = aligned_last + delta
    # renormalize to 1 (if >0)
    tot = w.sum()
    if tot <= 0:
        # fallback: equal
        w = pd.Series(1.0 / len(w), index=w.index)
    else:
        w = w / tot
    return w


def main():
    _ensure_reports()

    # Load metrics
    df = _load_metrics()
    df = _clip_dates(df)

    # Ensure only sleeves in our universe (drop others if present)
    df = df[df["sleeve"].isin(SLEEVES)].copy()
    if df.empty:
        # create 3 rows to avoid empty downstream
        df = _fallback_metrics()

    # Aggregate per-sleeve risk-adjusted score (bps space)
    rows = []
    mx_date = pd.to_datetime(df["date"].max())
    for s in SLEEVES:
        sub = df[df["sleeve"] == s].sort_values("date").tail(LOOKBACK_DAYS)
        mean_bps = _ewma(sub["ret_bps"], EWMA_ALPHA_MEAN) if not sub.empty else 0.0
        vol_bps = _ewma(sub["ret_bps"].abs(), EWMA_ALPHA_VOL) if not sub.empty else 1.0
        score = mean_bps / max(vol_bps, EPS)
        rows.append(
            {"sleeve": s, "mean_bps": mean_bps, "vol_bps": vol_bps, "score": score}
        )
    tab = pd.DataFrame(rows).set_index("sleeve")

    # Raw softmax allocation
    raw_w = pd.Series(
        _softmax(tab["score"].values, TEMPERATURE), index=tab.index, name="weight_raw"
    )

    # Apply safety caps & max-delta vs last policy
    last = _load_last_policy()
    fin_w = _apply_safety_caps(raw_w, last).rename("weight")
    fin_w = fin_w.round(6)

    # Compose policy table
    out = tab.copy()
    out["weight_raw"] = raw_w
    out["weight"] = fin_w
    out = out.reset_index().sort_values("weight", ascending=False)
    out["weight"] = out["weight"].round(6)
    out["weight_raw"] = out["weight_raw"].round(6)

    # Persist
    POLICY_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(POLICY_CSV, index=False)

    # Save last policy (for next week’s constraint)
    last_out = out[["sleeve", "weight"]].copy()
    last_out.to_csv(LAST_POLICY, index=False)

    # Summary JSON
    summary = {
        "as_of": str(mx_date.date()) if pd.notna(mx_date) else "",
        "sleeves": SLEEVES,
        "lookback_days": LOOKBACK_DAYS,
        "temperature": TEMPERATURE,
        "caps": {"min_w": MIN_W, "max_w": MAX_W, "max_delta": MAX_DELTA},
        "policy_csv": str(POLICY_CSV),
        "last_policy_csv": str(LAST_POLICY),
        "rows": int(out.shape[0]),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

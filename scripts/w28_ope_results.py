from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

"""
W28 — Off-Policy Evaluation (OPE)
Evaluates candidate target policies using logged bandit-like data with robust fallbacks.

Inputs (optional, auto-detected if present):
- reports/w27_last_policy.csv         # logging policy proxy (cols: sleeve, weight)
- reports/ic_timeseries.csv           # (date, sleeve, ic)
- reports/wk17_ops_attrib.csv         # (date, sleeve, ret or pnl)  -> ret_bps proxy
- reports/canary_log.csv              # (date, sleeve, realized_ret)-> ret_bps proxy

Fallbacks:
- Synthetic per-sleeve daily returns (ret_bps) and a synthetic logged trace (actions sampled from logging policy).

Outputs:
- reports/w28_ope_dataset.csv         # logged dataset used (date, action, pi_b, reward, plus sleeve ret_bps cols)
- reports/w28_ope_results.csv         # per policy x method: estimate + CI
- reports/w28_ope_summary.json        # knobs + file pointers
- reports/W28_review_*.zip            # via w28_build_review_zip.py

Methods:
- IPS (Inverse Propensity Scoring)
- SNIPS (Self-Normalized IPS)
- DM (Direct Method; model-free here uses full-information daily sleeve returns as oracle DM*)
- DR (Doubly Robust) using a simple Q-hat (EWMA mean per sleeve)

Note:
When full-information sleeve returns are available daily (as we construct here),
DM acts as an "oracle" diagnostic upper-bound. Real production can swap DM to a learned outcome model.
"""

ROOT = Path(r"F:\Projects\trading_stack_py")
REPORTS = ROOT / "reports"

# Optional inputs
LAST_POLICY = REPORTS / "w27_last_policy.csv"
IC_CSV = REPORTS / "ic_timeseries.csv"
ATTRIB_CSV = REPORTS / "wk17_ops_attrib.csv"
CANARY_CSV = REPORTS / "canary_log.csv"

# Outputs
DATASET_CSV = REPORTS / "w28_ope_dataset.csv"
RESULTS_CSV = REPORTS / "w28_ope_results.csv"
SUMMARY_JSON = REPORTS / "w28_ope_summary.json"

# Universe
SLEEVES = ["momentum_core", "event_drift", "shortterm_rev"]

# Knobs
LOOKBACK_DAYS = 60
RNG_SEED = 28
EWMA_ALPHA_Q = 0.20  # for Q-hat (per-sleeve EWMA mean)
N_BOOTSTRAP = 400  # CI resamples
BPS_TO_DEC = 1e-4
EPS = 1e-12


def _ensure_reports():
    REPORTS.mkdir(parents=True, exist_ok=True)


def _read_optional_csv(path: Path) -> pd.DataFrame | None:
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception:
        pass
    return None


def _ts_last_business_days(n=LOOKBACK_DAYS) -> pd.DatetimeIndex:
    return pd.date_range(end=pd.Timestamp.today().normalize(), periods=n, freq="B")


def _fallback_ret_bps() -> pd.DataFrame:
    idx = _ts_last_business_days(LOOKBACK_DAYS)
    rng = np.random.default_rng(RNG_SEED)
    base = {
        "momentum_core": rng.normal(3.0, 12.0, len(idx)),
        "event_drift": rng.normal(2.3, 10.0, len(idx)),
        "shortterm_rev": rng.normal(1.1, 15.0, len(idx)),
    }
    df = pd.DataFrame({"date": idx})
    for s, arr in base.items():
        df[s] = arr.astype(float)
    return df


def _load_ret_bps_table() -> pd.DataFrame:
    """
    Build a wide table with columns: date, momentum_core, event_drift, shortterm_rev (values in bps)
    Priority:
      (1) wk17_ops_attrib.csv -> map ret/pnl to bps (per sleeve)
      (2) ic_timeseries.csv   -> map IC to bps by scale
      (3) canary_log.csv      -> realized_ret to bps
      (4) fallback synthetic
    """
    # 1) attribution
    df = _read_optional_csv(ATTRIB_CSV)
    if df is not None and not df.empty:
        cols = {c.lower(): c for c in df.columns}
        dcol = cols.get("date")
        scol = cols.get("sleeve") or cols.get("bucket")
        rcol = cols.get("ret") or cols.get("return") or cols.get("pnl")
        if dcol and scol and rcol:
            tmp = df[[dcol, scol, rcol]].copy()
            tmp.columns = ["date", "sleeve", "val"]
            tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce")
            tmp["ret_bps"] = pd.to_numeric(tmp["val"], errors="coerce") * 1e4
            tmp = tmp.dropna(subset=["date", "sleeve", "ret_bps"])
            wide = tmp.pivot_table(
                index="date", columns="sleeve", values="ret_bps", aggfunc="last"
            )
            # keep only our sleeves
            wide = wide.reindex(columns=SLEEVES)
            wide = wide.reset_index()
            return wide

    # 2) ic_timeseries as proxy
    df = _read_optional_csv(IC_CSV)
    if df is not None and not df.empty:
        cols = {c.lower(): c for c in df.columns}
        dcol = cols.get("date")
        scol = cols.get("sleeve") or cols.get("signal") or cols.get("family")
        icol = cols.get("ic")
        if dcol and scol and icol:
            tmp = df[[dcol, scol, icol]].copy()
            tmp.columns = ["date", "sleeve", "ic"]
            tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce")
            tmp["ret_bps"] = pd.to_numeric(tmp["ic"], errors="coerce") * 12.0 * 10.0
            tmp = tmp.dropna(subset=["date", "sleeve", "ret_bps"])
            wide = tmp.pivot_table(
                index="date", columns="sleeve", values="ret_bps", aggfunc="last"
            )
            wide = wide.reindex(columns=SLEEVES)
            wide = wide.reset_index()
            return wide

    # 3) canary realized returns (if it exists for multiple sleeves)
    df = _read_optional_csv(CANARY_CSV)
    if df is not None and not df.empty:
        cols = {c.lower(): c for c in df.columns}
        dcol = cols.get("date")
        scol = cols.get("sleeve")
        rcol = cols.get("realized_ret") or cols.get("ret")
        if dcol and scol and rcol:
            tmp = df[[dcol, scol, rcol]].copy()
            tmp.columns = ["date", "sleeve", "val"]
            tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce")
            tmp["ret_bps"] = pd.to_numeric(tmp["val"], errors="coerce") * 1e4
            tmp = tmp.dropna(subset=["date", "sleeve", "ret_bps"])
            wide = tmp.pivot_table(
                index="date", columns="sleeve", values="ret_bps", aggfunc="last"
            )
            wide = wide.reindex(columns=SLEEVES)
            wide = wide.reset_index()
            return wide

    # 4) fallback synthetic
    return _fallback_ret_bps()


def _load_logging_policy() -> pd.Series:
    """
    Returns per-sleeve probabilities for the behavior/logging policy.
    If w27_last_policy.csv exists, use it; else uniform over SLEEVES.
    """
    if LAST_POLICY.exists():
        try:
            df = pd.read_csv(LAST_POLICY)
            cols = {c.lower(): c for c in df.columns}
            scol = cols.get("sleeve")
            wcol = cols.get("weight")
            if scol and wcol:
                s = pd.Series(
                    pd.to_numeric(df[wcol], errors="coerce").values,
                    index=df[scol].astype(str),
                )
                s = s.reindex(SLEEVES).fillna(0.0)
                tot = float(s.sum())
                if tot <= 0:
                    s = pd.Series(1.0 / len(SLEEVES), index=SLEEVES)
                else:
                    s = s / tot
                return s
        except Exception:
            pass
    return pd.Series(1.0 / len(SLEEVES), index=SLEEVES)


def _build_logged_dataset(ret_wide: pd.DataFrame, pi_b: pd.Series) -> pd.DataFrame:
    """
    Create logged dataset: one action per day sampled from logging policy.
    reward_t = ret_bps(date, action) converted to decimal.
    """
    df = ret_wide.copy()
    df = df.dropna(subset=["date"]).copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").tail(LOOKBACK_DAYS).reset_index(drop=True)
    rng = np.random.default_rng(RNG_SEED)
    acts = []
    probs = []
    rews = []
    for i, r in df.iterrows():
        # sample action from pi_b
        a_idx = rng.choice(len(SLEEVES), p=pi_b.values)
        a = SLEEVES[a_idx]
        acts.append(a)
        probs.append(float(pi_b[a]))
        # reward = that sleeve's ret in decimal
        rbps = (
            float(pd.to_numeric(r[a], errors="coerce"))
            if a in df.columns
            else float("nan")
        )
        rews.append(rbps * BPS_TO_DEC if math.isfinite(rbps) else 0.0)

    out = pd.DataFrame(
        {"date": df["date"], "action": acts, "pi_b": probs, "reward": rews}
    )
    # attach per-sleeve ret_bps columns for diagnostics / DM oracle
    for s in SLEEVES:
        if s in df.columns:
            out[f"ret_{s}_bps"] = pd.to_numeric(df[s], errors="coerce")
        else:
            out[f"ret_{s}_bps"] = np.nan
    return out


def _candidate_policies(pi_b: pd.Series) -> dict[str, pd.Series]:
    """
    Returns a dict of candidate evaluation policies pi_e (per-sleeve probabilities).
    Includes:
      - equal_weight
      - momentum_tilt (peaked on momentum_core)
      - last_policy (same as logging)
    """
    cands: dict[str, pd.Series] = {}
    # equal
    cands["equal_weight"] = pd.Series(1.0 / len(SLEEVES), index=SLEEVES)
    # momentum tilt
    w = pd.Series(1.0, index=SLEEVES)
    w[:] = 1.0
    w["momentum_core"] = 2.0
    w = w / w.sum()
    cands["momentum_tilt"] = w
    # last policy
    cands["last_policy"] = pi_b.copy()
    return cands


def _qhat_per_sleeve(dataset: pd.DataFrame) -> pd.Series:
    """
    Simple outcome model Q-hat(a) = EWMA mean reward for each sleeve using full-info columns.
    """
    q = {}
    for s in SLEEVES:
        col = f"ret_{s}_bps"
        if col in dataset.columns:
            series = pd.to_numeric(dataset[col], errors="coerce") * BPS_TO_DEC
            q[s] = (
                float(series.ewm(alpha=EWMA_ALPHA_Q).mean().iloc[-1])
                if series.notna().any()
                else 0.0
            )
        else:
            q[s] = 0.0
    return pd.Series(q)


def _ips(dataset: pd.DataFrame, pi_e: pd.Series) -> float:
    r = dataset["reward"].values
    a = dataset["action"].astype(str).values
    pib = dataset["pi_b"].values
    pie = np.array([float(pi_e.get(ai, 0.0)) for ai in a])
    w = np.where(pib > 0, pie / pib, 0.0)
    return float(np.mean(w * r))


def _snips(dataset: pd.DataFrame, pi_e: pd.Series) -> float:
    r = dataset["reward"].values
    a = dataset["action"].astype(str).values
    pib = dataset["pi_b"].values
    pie = np.array([float(pi_e.get(ai, 0.0)) for ai in a])
    w = np.where(pib > 0, pie / pib, 0.0)
    wsum = np.sum(w)
    if wsum <= 0:
        return 0.0
    return float(np.sum(w * r) / wsum)


def _dm_oracle_fullinfo(ret_wide_row: pd.Series, pi_e: pd.Series) -> float:
    """
    Oracle DM using full-information per-sleeve daily bps (converted to decimal).
    """
    val = 0.0
    for s, p in pi_e.items():
        col = f"ret_{s}_bps"
        rbps = float(pd.to_numeric(ret_wide_row.get(col, np.nan), errors="coerce"))
        r = (rbps * BPS_TO_DEC) if math.isfinite(rbps) else 0.0
        val += p * r
    return val


def _dr(dataset: pd.DataFrame, pi_e: pd.Series, qhat: pd.Series) -> float:
    """
    DR = average_t [ sum_a pi_e(a|x_t) Qhat(a,x_t)  +  (pi_e(a_t|x_t)/pi_b(a_t|x_t)) * (r_t - Qhat(a_t,x_t)) ]
    Here Qhat(a,x) is constant per sleeve (EWMA mean) for simplicity.
    """
    # term1: policy value under model
    term1 = 0.0
    for s, p in pi_e.items():
        term1 += p * float(qhat.get(s, 0.0))
    # term2: correction
    a = dataset["action"].astype(str).values
    r = dataset["reward"].values
    pib = dataset["pi_b"].values
    pie = np.array([float(pi_e.get(ai, 0.0)) for ai in a])
    q_at = np.array([float(qhat.get(ai, 0.0)) for ai in a])
    w = np.where(pib > 0, pie / pib, 0.0)
    corr = w * (r - q_at)
    return float(term1 + np.mean(corr))


def _bootstrap(
    dataset: pd.DataFrame,
    pi_e: pd.Series,
    method: str,
    qhat: pd.Series | None,
    n=N_BOOTSTRAP,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(RNG_SEED)
    vals = []
    nrows = len(dataset)
    for _ in range(n):
        idx = rng.integers(0, nrows, size=nrows)
        samp = dataset.iloc[idx]
        if method == "IPS":
            v = _ips(samp, pi_e)
        elif method == "SNIPS":
            v = _snips(samp, pi_e)
        elif method == "DM":
            v = float(
                np.mean([_dm_oracle_fullinfo(row, pi_e) for _, row in samp.iterrows()])
            )
        elif method == "DR":
            v = _dr(
                samp, pi_e, qhat if qhat is not None else pd.Series(0.0, index=SLEEVES)
            )
        else:
            v = 0.0
        vals.append(v)
    arr = np.array(vals)
    mean = float(arr.mean())
    lo, hi = np.percentile(arr, [2.5, 97.5])
    return mean, float(lo), float(hi)


def _annualize(p_decimal_per_day: float) -> float:
    # rough compounding to annual pct: (1 + r_d)^{252} - 1
    return (1.0 + p_decimal_per_day) ** 252 - 1.0


def main():
    _ensure_reports()

    # 1) build per-sleeve daily ret table (bps)
    ret_wide = _load_ret_bps_table()
    # sanity window
    ret_wide = ret_wide.dropna(subset=["date"]).copy()
    ret_wide["date"] = pd.to_datetime(ret_wide["date"], errors="coerce")
    ret_wide = ret_wide.sort_values("date").tail(LOOKBACK_DAYS).reset_index(drop=True)
    # fill missing sleeves with 0 bps (conservative) to keep pipeline smooth
    for s in SLEEVES:
        if s not in ret_wide.columns:
            ret_wide[s] = 0.0

    # 2) logging policy (behavior)
    pi_b = _load_logging_policy()

    # 3) synthesize logged dataset (action per day, reward of chosen action)
    ds = _build_logged_dataset(ret_wide, pi_b)

    # 4) candidate target policies
    candidates = _candidate_policies(pi_b)

    # 5) Q-hat
    qhat = _qhat_per_sleeve(ds)

    # 6) Evaluate
    rows = []
    for name, pi_e in candidates.items():
        # IPS / SNIPS
        ips_mean, ips_lo, ips_hi = _bootstrap(ds, pi_e, "IPS", qhat)
        snips_mean, snips_lo, snips_hi = _bootstrap(ds, pi_e, "SNIPS", qhat)
        # DM (oracle full-info, diagnostic)
        dm_mean, dm_lo, dm_hi = _bootstrap(ds, pi_e, "DM", qhat)
        # DR
        dr_mean, dr_lo, dr_hi = _bootstrap(ds, pi_e, "DR", qhat)

        for method, mean, lo, hi in [
            ("IPS", ips_mean, ips_lo, ips_hi),
            ("SNIPS", snips_mean, snips_lo, snips_hi),
            ("DM_oracle", dm_mean, dm_lo, dm_hi),
            ("DR", dr_mean, dr_lo, dr_hi),
        ]:
            rows.append(
                {
                    "policy": name,
                    "method": method,
                    "estimate_bps_per_day": round(mean * 1e4, 3),
                    "ci_lo_bps_per_day": round(lo * 1e4, 3),
                    "ci_hi_bps_per_day": round(hi * 1e4, 3),
                    "approx_annual_pct": round(_annualize(mean) * 100.0, 2),
                }
            )

    # 7) Persist artifacts
    # Attach full-info sleeve bps columns to dataset for reference
    for s in SLEEVES:
        if f"ret_{s}_bps" not in ds.columns and s in ret_wide.columns:
            # align by date
            ds = ds.merge(
                ret_wide[["date", s]].rename(columns={s: f"ret_{s}_bps"}),
                on="date",
                how="left",
            )

    DATASET_CSV.parent.mkdir(parents=True, exist_ok=True)
    ds.to_csv(DATASET_CSV, index=False)

    res = pd.DataFrame(rows)
    res.to_csv(RESULTS_CSV, index=False)

    summary = {
        "as_of": (
            str(pd.to_datetime(ret_wide["date"].max()).date())
            if not ret_wide.empty
            else ""
        ),
        "sleeves": SLEEVES,
        "lookback_days": LOOKBACK_DAYS,
        "bootstrap": N_BOOTSTRAP,
        "dataset_csv": str(DATASET_CSV),
        "results_csv": str(RESULTS_CSV),
        "logging_policy": dict(pi_b),
        "qhat": dict(qhat),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

"""
W29 — Safe Policy Improvement (SPIBB-style, bandit simplification)

Inputs (from W28):
- reports/w28_ope_dataset.csv     # columns: date, action, pi_b, reward, ret_<sleeve>_bps...
- reports/w28_ope_summary.json    # logging policy + Q-hat (EWMA) + sleeves list

Optional (from W27):
- reports/w27_last_policy.csv     # prior (logging) policy proxy: sleeve, weight

Outputs:
- reports/w29_safe_policy.csv     # proposed safe policy weights per sleeve
- reports/w29_safety_diag.json    # diagnostics: support counts, constraints, improvement ests
- reports/w29_policy_eval.csv     # value estimates (DR/IPS/SNIPS) for baseline vs safe

Method (bandit SPIBB intuition):
- Identify "supported" actions A_supp = {a: N_b(a) >= N_MIN}.
- Start from logging policy π_b. Reallocate at most EPS_L1 mass from unsupported actions
  toward the best supported actions by Q-hat, capping total L1 drift ≤ L1_CAP and
  never increasing unsupported action probabilities.
- Accept new policy only if DR estimate improves and bootstrap CI lower bound ≥ 0 (non-negative gain).
- Else, fallback to π_b.

Notes:
- This is a safe-conservative variant, not the full SPIBB theorem; tuned for small action set (sleeves).
"""

ROOT = Path(r"F:\Projects\trading_stack_py")
REPORTS = ROOT / "reports"
DATASET = REPORTS / "w28_ope_dataset.csv"
SUMMARY = REPORTS / "w28_ope_summary.json"
LAST_POL = REPORTS / "w27_last_policy.csv"

OUT_POLICY = REPORTS / "w29_safe_policy.csv"
OUT_EVAL = REPORTS / "w29_policy_eval.csv"
OUT_DIAG = REPORTS / "w29_safety_diag.json"

# Knobs
N_MIN_VISITS = 10  # minimum behavior visits for "supported"
L1_CAP = 0.30  # total allowed L1 distance from π_b
EPS_L1_SHIFT = 0.20  # how much prob mass we try to move (capped by L1_CAP)
N_BOOTSTRAP = 400
RNG_SEED = 29

BPS_TO_DEC = 1e-4


def _read_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_last_policy(sleeves: list[str]) -> pd.Series | None:
    if LAST_POL.exists():
        try:
            df = pd.read_csv(LAST_POL)
            cols = {c.lower(): c for c in df.columns}
            scol = cols.get("sleeve")
            wcol = cols.get("weight")
            if scol and wcol:
                s = pd.Series(
                    pd.to_numeric(df[wcol], errors="coerce").values,
                    index=df[scol].astype(str),
                )
                s = s.reindex(sleeves).fillna(0.0)
                tot = float(s.sum())
                if tot > 0:
                    return s / tot
        except Exception:
            pass
    return None


def _sn_to_series(obj: dict, sleeves: list[str]) -> pd.Series:
    s = pd.Series({k: float(v) for k, v in obj.items()})
    s = s.reindex(sleeves).fillna(0.0)
    tot = float(s.sum())
    return s / tot if tot > 0 else pd.Series(1.0 / len(sleeves), index=sleeves)


def _load_inputs():
    if not DATASET.exists():
        raise FileNotFoundError(f"Missing {DATASET}; run W28 first.")
    ds = pd.read_csv(DATASET)
    sumj = _read_json(SUMMARY) if SUMMARY.exists() else {}

    sleeves = sumj.get("sleeves", ["momentum_core", "event_drift", "shortterm_rev"])
    qhat = _sn_to_series(sumj.get("qhat", {}), sleeves)
    pi_b_summary = sumj.get("logging_policy", {})
    pi_b = _sn_to_series(pi_b_summary, sleeves)

    # If W27 last policy exists (more precise logging proxy), prefer it
    last = _read_last_policy(sleeves)
    if last is not None and not np.isclose(last.sum(), 0.0):
        pi_b = last / last.sum()

    # Sanitize dataset
    ds = ds.copy()
    if "action" not in ds or "pi_b" not in ds or "reward" not in ds:
        raise ValueError(
            "w28_ope_dataset.csv missing required columns: action, pi_b, reward"
        )
    ds["action"] = ds["action"].astype(str).str.strip()
    ds["pi_b"] = pd.to_numeric(ds["pi_b"], errors="coerce").fillna(0.0)
    ds["reward"] = pd.to_numeric(ds["reward"], errors="coerce").fillna(0.0)
    ds = ds[ds["action"].isin(sleeves)].reset_index(drop=True)

    return sleeves, qhat, pi_b, ds


def _dr_value(ds: pd.DataFrame, pi_e: pd.Series, qhat: pd.Series) -> float:
    # DR with constant Q-hat(a)
    term1 = float(np.sum(pi_e.reindex(qhat.index).fillna(0.0) * qhat))
    a = ds["action"].astype(str).values
    r = ds["reward"].values
    pib = ds["pi_b"].values
    pie = np.array([float(pi_e.get(ai, 0.0)) for ai in a])
    q_at = np.array([float(qhat.get(ai, 0.0)) for ai in a])
    w = np.where(pib > 0, pie / pib, 0.0)
    return float(term1 + np.mean(w * (r - q_at)))


def _ips_value(ds: pd.DataFrame, pi_e: pd.Series) -> float:
    a = ds["action"].astype(str).values
    r = ds["reward"].values
    pib = ds["pi_b"].values
    pie = np.array([float(pi_e.get(ai, 0.0)) for ai in a])
    w = np.where(pib > 0, pie / pib, 0.0)
    return float(np.mean(w * r))


def _snips_value(ds: pd.DataFrame, pi_e: pd.Series) -> float:
    a = ds["action"].astype(str).values
    r = ds["reward"].values
    pib = ds["pi_b"].values
    pie = np.array([float(pi_e.get(ai, 0.0)) for ai in a])
    w = np.where(pib > 0, pie / pib, 0.0)
    wsum = float(np.sum(w))
    if wsum <= 0:
        return 0.0
    return float(np.sum(w * r) / wsum)


def _bootstrap_gain(
    ds: pd.DataFrame,
    pi_b: pd.Series,
    pi_cand: pd.Series,
    qhat: pd.Series,
    n=N_BOOTSTRAP,
    seed=RNG_SEED,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    nrows = len(ds)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, nrows, size=nrows)
        samp = ds.iloc[idx]
        v_new = _dr_value(samp, pi_cand, qhat)
        v_old = _dr_value(samp, pi_b, qhat)
        vals.append(v_new - v_old)
    arr = np.array(vals, float)
    mean = float(arr.mean())
    lo, hi = np.percentile(arr, [2.5, 97.5])
    return mean, float(lo), float(hi)


def _counts_by_action(ds: pd.DataFrame, sleeves: list[str]) -> pd.Series:
    cnt = ds["action"].value_counts().reindex(sleeves).fillna(0).astype(int)
    return cnt


def _project_simplex(v: np.ndarray) -> np.ndarray:
    """Project vector v onto probability simplex."""
    # Algorithm from Duchi et al. (2008)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(u) + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1.0)
    w = np.maximum(v - theta, 0.0)
    # numerical fix
    s = w.sum()
    if s <= 0:
        w = np.ones_like(w) / len(w)
    else:
        w = w / s
    return w


def _l1_distance(p: pd.Series, q: pd.Series) -> float:
    s = p.index
    return float(
        np.abs(p.reindex(s).fillna(0.0).values - q.reindex(s).fillna(0.0).values).sum()
    )


def _propose_safe_policy(
    sleeves: list[str], pi_b: pd.Series, qhat: pd.Series, counts: pd.Series
) -> pd.Series:
    """
    Start from π_b. Identify supported set S = {a: N_b(a) >= N_MIN_VISITS}.
    - For unsupported U: enforce π_e(a) <= π_b(a).
    - Shift up to EPS_L1_SHIFT mass from U toward the best in S by qhat, cap total L1 ≤ L1_CAP.
    - Re-project to simplex and clamp constraints.
    """
    p = pi_b.copy().reindex(sleeves).fillna(0.0)
    p = p / float(p.sum() or 1.0)

    supported = counts[counts >= N_MIN_VISITS].index.tolist()
    unsupported = [a for a in sleeves if a not in supported]

    # If nothing is supported, return π_b (no safe improvement possible)
    if len(supported) == 0:
        return p

    # Determine candidate shift direction: move mass to best supported by qhat
    q_sorted = qhat.reindex(supported).fillna(0.0).sort_values(ascending=False)
    if q_sorted.empty:
        return p
    best = q_sorted.index[0]

    # Max shift allowed by L1_CAP (remember L1 distance counts both give & take)
    max_shift = min(EPS_L1_SHIFT, L1_CAP / 2.0)
    if max_shift <= 0:
        return p

    # Create raw vector v
    v = p.copy()
    # Reduce mass from unsupported first (cannot increase them)
    take_total = 0.0
    for a in unsupported:
        can_take = min(v[a], max_shift - take_total)
        if can_take > 0:
            v[a] -= can_take
            take_total += can_take
        if take_total >= max_shift:
            break

    # If still room, gently reduce other supported-but-not-best to free mass
    if take_total < max_shift:
        for a in supported:
            if a == best:
                continue
            can_take = min(v[a], max_shift - take_total) * 0.5  # small nudge
            if can_take > 0:
                v[a] -= can_take
                take_total += can_take
            if take_total >= max_shift:
                break

    # Add all taken mass to best supported action
    v[best] += take_total

    # Project to simplex
    w = pd.Series(_project_simplex(v.values), index=v.index)

    # Enforce "no increase" on unsupported actions
    for a in unsupported:
        if w[a] > p[a]:
            # clamp back to p[a], re-distribute the excess to supported (favor best)
            excess = w[a] - p[a]
            w[a] = p[a]
            # redistribute excess to supported (weighted by qhat rank)
            denom = q_sorted.clip(lower=0).sum()
            if denom <= 0:
                denom = float(len(supported))
                share = excess / denom
                for s in supported:
                    w[s] += share
            else:
                for s in supported:
                    w[s] += (
                        float(excess) * float(max(qhat.get(s, 0.0), 0.0)) / float(denom)
                    )

    # Final L1 cap
    if _l1_distance(p, w) > L1_CAP:
        # Move w towards p to satisfy cap
        alpha = L1_CAP / _l1_distance(p, w)
        w = p + alpha * (w - p)
        w = pd.Series(_project_simplex(w.values), index=w.index)

    # Ensure numerics
    w = w.clip(lower=0.0)
    w = w / float(w.sum() or 1.0)
    return w


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    sleeves, qhat, pi_b, ds = _load_inputs()

    # Support counts under behavior policy
    counts = _counts_by_action(ds, sleeves)

    # Baseline value
    v_b_dr = _dr_value(ds, pi_b, qhat)
    v_b_ips = _ips_value(ds, pi_b)
    v_b_snips = _snips_value(ds, pi_b)

    # Candidate safe policy
    pi_cand = _propose_safe_policy(sleeves, pi_b, qhat, counts)

    # Evaluate candidate
    v_c_dr = _dr_value(ds, pi_cand, qhat)
    v_c_ips = _ips_value(ds, pi_cand)
    v_c_snips = _snips_value(ds, pi_cand)

    # Bootstrap DR gain
    gain_mean, gain_lo, gain_hi = _bootstrap_gain(
        ds, pi_b, pi_cand, qhat, n=N_BOOTSTRAP, seed=RNG_SEED
    )

    # Accept only if CI lower bound >= 0 (non-negative improvement)
    accepted = gain_lo >= 0.0
    pi_out = pi_cand if accepted else pi_b.copy()

    # Save outputs
    out_policy = pd.DataFrame(
        {
            "sleeve": sleeves,
            "weight": [float(pi_out.get(s, 0.0)) for s in sleeves],
            "source": ["accepted" if accepted else "baseline"][0],
        }
    )
    OUT_POLICY.parent.mkdir(parents=True, exist_ok=True)
    out_policy.to_csv(OUT_POLICY, index=False)

    eval_rows = [
        {
            "policy": "baseline",
            "method": "DR",
            "est_per_day_bps": round(v_b_dr * 1e4, 3),
        },
        {
            "policy": "baseline",
            "method": "IPS",
            "est_per_day_bps": round(v_b_ips * 1e4, 3),
        },
        {
            "policy": "baseline",
            "method": "SNIPS",
            "est_per_day_bps": round(v_b_snips * 1e4, 3),
        },
        {
            "policy": "candidate",
            "method": "DR",
            "est_per_day_bps": round(v_c_dr * 1e4, 3),
        },
        {
            "policy": "candidate",
            "method": "IPS",
            "est_per_day_bps": round(v_c_ips * 1e4, 3),
        },
        {
            "policy": "candidate",
            "method": "SNIPS",
            "est_per_day_bps": round(v_c_snips * 1e4, 3),
        },
    ]
    pd.DataFrame(eval_rows).to_csv(OUT_EVAL, index=False)

    diag = {
        "sleeves": sleeves,
        "counts": {k: int(v) for k, v in counts.items()},
        "supported_min": N_MIN_VISITS,
        "l1_cap": L1_CAP,
        "l1_distance": round(_l1_distance(pi_b, pi_cand), 6),
        "accepted": bool(accepted),
        "gain_dr_per_day_bps": round(gain_mean * 1e4, 3),
        "gain_dr_ci_bps": [round(gain_lo * 1e4, 3), round(gain_hi * 1e4, 3)],
        "pi_b": {k: float(pi_b.get(k, 0.0)) for k in sleeves},
        "pi_cand": {k: float(pi_cand.get(k, 0.0)) for k in sleeves},
        "pi_out": {k: float(pi_out.get(k, 0.0)) for k in sleeves},
    }
    OUT_DIAG.write_text(json.dumps(diag, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "policy_csv": str(OUT_POLICY),
                "eval_csv": str(OUT_EVAL),
                "diag_json": str(OUT_DIAG),
                "accepted": accepted,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

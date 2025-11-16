from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

"""
W30 — Risk-Sensitive Objectives
Objective: maximize  E[r(w)]  − λ·Var[r(w)]  − γ·SemiVar[r(w)]
subject to: w >= 0, sum(w)=1, L1(w, w_baseline) <= L1_CAP

Inputs:
- reports/w28_ope_dataset.csv   # must contain per-action daily return columns: ret_<sleeve>_bps
- reports/w29_safe_policy.csv   # baseline policy (weights per sleeve)
- (optional) reports/w27_last_policy.csv  # used only if w29 missing

Outputs:
- reports/w30_risk_sens_policy.csv        # best policy (weights)
- reports/w30_risk_sens_curve.csv         # grid sweep metrics per (lambda, gamma)
- reports/w30_risk_sens_diag.json         # diagnostics, knobs, acceptance decision
- reports/w30_policy_eval.csv             # realized backtest-like stats vs baseline (in-sample)
"""

ROOT = Path(r"F:\Projects\trading_stack_py")
REPORTS = ROOT / "reports"

DATASET = REPORTS / "w28_ope_dataset.csv"
SAFE_POL_CSV = REPORTS / "w29_safe_policy.csv"
LAST_POL_CSV = REPORTS / "w27_last_policy.csv"

OUT_WEIGHTS = REPORTS / "w30_risk_sens_policy.csv"
OUT_CURVE = REPORTS / "w30_risk_sens_curve.csv"
OUT_DIAG = REPORTS / "w30_risk_sens_diag.json"
OUT_PEV = REPORTS / "w30_policy_eval.csv"

# ---- knobs ----
LAMBDAS = [0.25, 0.5, 1.0, 2.0, 4.0]  # variance aversion
GAMMAS = [0.0, 0.25, 0.5, 1.0]  # downside semivariance penalty
N_STEPS = 500  # projected gradient iterations
LR = 0.5  # base learning rate (adaptive)
L1_CAP = 0.30  # max L1 drift from baseline
SEED = 30

BPS = 1e-4  # bps -> decimal


# ------------- utilities -------------
def _project_simplex(v: np.ndarray) -> np.ndarray:
    """Project onto nonnegative simplex {w: w>=0, sum=1}."""
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * (np.arange(1, len(u) + 1)) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1.0)
    w = np.maximum(v - theta, 0.0)
    s = w.sum()
    return w / s if s > 0 else np.ones_like(w) / len(w)


def _l1(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.abs(p - q).sum())


def _clamp_l1(w: np.ndarray, w0: np.ndarray, cap: float) -> np.ndarray:
    d = _l1(w, w0)
    if d <= cap + 1e-12:
        return w
    # move w towards w0 to satisfy cap
    alpha = cap / d
    w2 = w0 + alpha * (w - w0)
    return _project_simplex(w2)


def _load_baseline(sleeves: list[str]) -> pd.Series:
    if SAFE_POL_CSV.exists():
        df = pd.read_csv(SAFE_POL_CSV)
    elif LAST_POL_CSV.exists():
        df = pd.read_csv(LAST_POL_CSV)
    else:
        # equal weights
        return pd.Series(1.0 / len(sleeves), index=sleeves)

    cols = {c.lower(): c for c in df.columns}
    scol = cols.get("sleeve")
    wcol = cols.get("weight")
    if scol and wcol:
        s = pd.Series(pd.to_numeric(df[wcol], errors="coerce").values, index=df[scol].astype(str))
        s = s.reindex(sleeves).fillna(0.0)
        tot = float(s.sum())
        return s / tot if tot > 0 else pd.Series(1.0 / len(sleeves), index=sleeves)
    return pd.Series(1.0 / len(sleeves), index=sleeves)


def _load_dataset():
    if not DATASET.exists():
        raise FileNotFoundError(f"Missing {DATASET}; run W28 first.")
    ds = pd.read_csv(DATASET)
    # infer sleeve columns as ret_<sleeve>_bps *
    sleeves = []
    for c in ds.columns:
        cl = c.lower()
        if cl.startswith("ret_") and cl.endswith("_bps"):
            sleeves.append(c[len("ret_") : -len("_bps")])
    sleeves = sorted(set(sleeves))
    if not sleeves:
        raise ValueError("No ret_<sleeve>_bps columns found in w28_ope_dataset.csv")

    # Build returns matrix (T x K) in decimal/day
    R = []
    for s in sleeves:
        col = f"ret_{s}_bps"
        R.append(pd.to_numeric(ds[col], errors="coerce").fillna(0.0).values * BPS)
    R = np.stack(R, axis=1)  # T x K
    return ds, sleeves, R


def _metrics_from_path(rt: np.ndarray) -> dict:
    """Compute daily mean, var, semivar, cvar95 (decimal)."""
    m = float(np.nanmean(rt))
    v = float(np.nanvar(rt, ddof=1)) if len(rt) > 1 else 0.0
    neg = np.minimum(rt, 0.0)
    semi = float(np.nanmean(neg**2))
    # CVaR 95% (average of worst 5% tail); if too short, fallback to min-tail mean
    q = np.nanpercentile(rt, 5.0)
    tail = rt[rt <= q]
    cvar = float(np.nanmean(tail)) if tail.size > 0 else float(np.nanmin(rt) if rt.size else 0.0)
    return {"mean": m, "var": v, "semivar": semi, "cvar95": cvar}


def _objective_and_grad(w: np.ndarray, R: np.ndarray, lam: float, gam: float) -> tuple[float, np.ndarray]:
    """
    r_t = (R @ w)
    f = mean(r_t) - lam * var(r_t) - gam * semivar(r_t)
    grad = d/dw above. semivar uses piecewise grad for r_t<0.
    """
    rt = R @ w  # T
    T = float(len(rt)) if len(rt) else 1.0
    mean = float(rt.mean())
    # variance and grad
    m_t = mean
    diff = rt - m_t
    var = float((diff @ diff) / max(len(rt) - 1, 1))
    # grad mean: mean(R, axis=0)
    g_mean = R.mean(axis=0)
    # grad var: d Var[r] / dw = 2 * Cov(Rw, R)  (sample version)
    # Approx via: 2/T * R^T (rt - mean)
    g_var = (2.0 / max(T - 1.0, 1.0)) * (R.T @ (rt - rt.mean()))
    # semivar grad: mean( (2*neg) * R_t ) where neg = min(0, rt)
    neg = np.minimum(rt, 0.0)
    g_semi = (2.0 / T) * (R.T @ neg)

    f = mean - lam * var - gam * float(np.mean(neg**2))
    g = g_mean - lam * g_var - gam * g_semi
    return f, g


def _optimize(w0: np.ndarray, R: np.ndarray, lam: float, gam: float, l1_cap: float) -> np.ndarray:
    w = w0.copy()
    lr = LR
    best_w, best_f = w.copy(), -1e99
    for it in range(N_STEPS):
        f, g = _objective_and_grad(w, R, lam, gam)
        # record
        if f > best_f:
            best_f, best_w = f, w.copy()
        # step with backtracking
        step = lr / math.sqrt(1.0 + it)
        cand = w + step * g
        cand = _project_simplex(cand)
        cand = _clamp_l1(cand, w0, l1_cap)
        w = cand
    return best_w


def main():
    np.random.seed(SEED)
    REPORTS.mkdir(parents=True, exist_ok=True)

    ds, sleeves, R = _load_dataset()
    w_baseline = _load_baseline(sleeves)
    w0 = w_baseline.reindex(sleeves).fillna(0.0).values
    w0 = _project_simplex(w0)

    # Evaluate baseline
    rt_base = R @ w0
    base_mets = _metrics_from_path(rt_base)
    base_row = {"policy": "baseline", **{k: round(v, 8) for k, v in base_mets.items()}}

    # Sweep grid
    rows = []
    best_score = -1e99
    best = {"w": w0, "lam": None, "gam": None, "mets": base_mets, "score": -1e99}
    for lam in LAMBDAS:
        for gam in GAMMAS:
            w = _optimize(w0, R, lam, gam, L1_CAP)
            rt = R @ w
            mets = _metrics_from_path(rt)
            # scalarization for comparison (risk-adjusted daily): mean − lam*var − gam*semivar
            score = mets["mean"] - lam * mets["var"] - gam * mets["semivar"]
            rows.append(
                {
                    "lambda": lam,
                    "gamma": gam,
                    "mean": round(mets["mean"], 8),
                    "var": round(mets["var"], 8),
                    "semivar": round(mets["semivar"], 8),
                    "cvar95": round(mets["cvar95"], 8),
                    "l1_from_baseline": round(_l1(w, w0), 6),
                    "score": round(score, 10),
                    **{f"w_{s}": round(float(w[i]), 6) for i, s in enumerate(sleeves)},
                }
            )
            if score > best_score:
                best_score = score
                best = {
                    "w": w.copy(),
                    "lam": lam,
                    "gam": gam,
                    "mets": mets,
                    "score": score,
                }

    # Compare best vs baseline; accept if improvement in scalarized score and better (>=) CVaR95
    accept = best_score > (base_mets["mean"] - LAMBDAS[0] * base_mets["var"] - GAMMAS[0] * base_mets["semivar"])
    accept = accept and (best["mets"]["cvar95"] >= base_mets["cvar95"])  # no worse left-tail

    w_out = best["w"] if accept else w0
    src = "accepted" if accept else "baseline"

    # Save outputs
    curve_df = pd.DataFrame(rows)
    curve_df.to_csv(OUT_CURVE, index=False)

    weights_df = pd.DataFrame(
        {
            "sleeve": sleeves,
            "weight": [float(w_out[i]) for i in range(len(sleeves))],
            "source": src,
        }
    )
    weights_df.to_csv(OUT_WEIGHTS, index=False)

    # policy_eval style table
    eval_rows = [
        {
            "policy": "baseline",
            "method": "path",
            "mean_bps": round(base_mets["mean"] / BPS, 3),
            "var": round(base_mets["var"], 8),
            "semivar": round(base_mets["semivar"], 8),
            "cvar95_bps": round(base_mets["cvar95"] / BPS, 3),
        },
        {
            "policy": "candidate",
            "method": "path",
            "mean_bps": round(best["mets"]["mean"] / BPS, 3),
            "var": round(best["mets"]["var"], 8),
            "semivar": round(best["mets"]["semivar"], 8),
            "cvar95_bps": round(best["mets"]["cvar95"] / BPS, 3),
        },
    ]
    pd.DataFrame(eval_rows).to_csv(OUT_PEV, index=False)

    diag = {
        "sleeves": sleeves,
        "lambdas": LAMBDAS,
        "gammas": GAMMAS,
        "l1_cap": L1_CAP,
        "accepted": bool(accept),
        "chosen_lambda": best["lam"] if accept else None,
        "chosen_gamma": best["gam"] if accept else None,
        "baseline_metrics": base_mets,
        "candidate_metrics": best["mets"],
        "notes": "Downside uses semivariance on weighted path; acceptance requires non-worse CVaR95 and higher scalar score.",
    }
    OUT_DIAG.write_text(json.dumps(diag, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "weights_csv": str(OUT_WEIGHTS),
                "curve_csv": str(OUT_CURVE),
                "diag_json": str(OUT_DIAG),
                "policy_eval_csv": str(OUT_PEV),
                "accepted": bool(accept),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

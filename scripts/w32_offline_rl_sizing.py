from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

"""
W32 — Offline RL for Sizing (safe policy, offline only)

Idea (simple, robust):
- Build offline dataset from past (date,ticker,target features) -> realized reward proxy.
- Reward proxy = forward 5d return (from prices) - execution cost (bps/1e4).
- Fit a tiny ridge regression y ~ X (offline policy evaluation-lite).
- Policy: size factor f = sigmoid(gain * (w^T x_norm)), then
          w_rl = clip(f * target_w * kelly_boost, 0, per_name_cap), observing sector/cap notes later.
- Hard caps & Kelly fraction keep it safe (no live orders here).

Inputs:
- reports/wk11_blend_targets.csv  [date,ticker,base_w,final_mult,target_w]  (already in your repo)
- data/prices/<TICKER>.parquet     [date, close] used to compute fwd returns
- reports/wk13_dryrun_fills.csv    [ticker, slippage_bps, commission_bps, tax_bps, notional_ref] (optional costs)
Outputs:
- reports/w32_offline_rl_sizing.csv  [date,ticker,target_w,size_factor,w_rl,notes]
- reports/w32_offline_rl_diag.json   summary, coefficients, coverage
"""

ROOT = Path(r"F:\Projects\trading_stack_py")
REPORTS = ROOT / "reports"
DATA = ROOT / "data" / "prices"

TARGETS_CSV = REPORTS / "wk11_blend_targets.csv"
FILLS_CSV = REPORTS / "wk13_dryrun_fills.csv"

OUT_CSV = REPORTS / "w32_offline_rl_sizing.csv"
OUT_JSON = REPORTS / "w32_offline_rl_diag.json"

# ---- knobs (tunable, safe defaults) ----
HORIZON_DAYS = 5  # forward return horizon (trading days preferred; date skip fallback)
RIDGE_L2 = 1e-3  # ridge penalty
SIGMOID_GAIN = 6.0  # steeper -> more decisive sizing
KELLY_FRACTION = 0.25  # fractional kelly cap multiplier
PER_NAME_CAP = 0.06  # 6% max per name (base profile)
SECTOR_CAP_PLACEHOLDER = 0.35  # not enforced here; documented for allocator
DEFAULT_COST_BPS = 20.0  # when no fills: assume 20 bps bps drag
MAX_TRAIN_ROWS = 100_000  # just in case


def _safe_read_csv(p: Path) -> pd.DataFrame | None:
    try:
        if p.exists():
            return pd.read_csv(p)
    except Exception:
        pass
    return None


def _load_targets() -> pd.DataFrame:
    df = _safe_read_csv(TARGETS_CSV)
    if df is None or df.empty:
        raise FileNotFoundError(f"Missing or empty {TARGETS_CSV}. Run W11 first.")
    cols = {c.lower(): c for c in df.columns}
    need = ["date", "ticker", "base_w", "final_mult", "target_w"]
    for n in need:
        if cols.get(n) is None:
            raise ValueError(f"{TARGETS_CSV} missing column: {n}")
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df[cols["date"]], errors="coerce"),
            "ticker": df[cols["ticker"]].astype(str),
            "base_w": pd.to_numeric(df[cols["base_w"]], errors="coerce"),
            "final_mult": pd.to_numeric(df[cols["final_mult"]], errors="coerce"),
            "target_w": pd.to_numeric(df[cols["target_w"]], errors="coerce"),
        }
    ).dropna()
    # Keep sensible ranges
    out = out[(out["target_w"] >= 0) & (out["target_w"] <= 1.0)]
    return out


def _load_cost_bps_map() -> dict[str, float]:
    df = _safe_read_csv(FILLS_CSV)
    if df is None or df.empty:  # default cost if nothing there
        return {}
    cols = {c.lower(): c for c in df.columns}
    sb = cols.get("slippage_bps")
    cb = cols.get("commission_bps")
    tb = cols.get("tax_bps")
    tic = cols.get("ticker")
    if tic is None or sb is None:
        return {}
    s = pd.to_numeric(df[sb], errors="coerce").fillna(DEFAULT_COST_BPS)
    c = pd.to_numeric(df[cb], errors="coerce").fillna(0.0) if cb else 0.0
    t = pd.to_numeric(df[tb], errors="coerce").fillna(0.0) if tb else 0.0
    cost = s + (c if np.isscalar(c) else c) + (t if np.isscalar(t) else t)
    g = pd.DataFrame({"ticker": df[tic].astype(str), "cost": cost})
    m = g.groupby("ticker", as_index=False)["cost"].median()
    return dict(zip(m["ticker"], m["cost"], strict=False))


def _read_price(ticker: str) -> pd.DataFrame | None:
    p = DATA / f"{ticker}.parquet"
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p)
        cols = {c.lower(): c for c in df.columns}
        dcol = cols.get("date") or cols.get("dt")
        ccol = cols.get("close") or cols.get("px_close") or cols.get("price")
        if not dcol or not ccol:
            return None
        out = pd.DataFrame(
            {
                "date": pd.to_datetime(df[dcol], errors="coerce"),
                "close": pd.to_numeric(df[ccol], errors="coerce"),
            }
        ).dropna()
        out = out.sort_values("date").reset_index(drop=True)
        return out
    except Exception:
        return None


def _fwd_return(price_df: pd.DataFrame, d: pd.Timestamp, horizon_days: int) -> float | None:
    if price_df is None or price_df.empty:
        return None
    # find index of date d (if absent, use next available after d)
    idx = price_df["date"].searchsorted(d, side="left")
    if idx >= len(price_df):  # after last date
        return None
    px0 = float(price_df.iloc[idx]["close"])
    j = idx + horizon_days
    if j >= len(price_df):
        return None
    px1 = float(price_df.iloc[j]["close"])
    if not (math.isfinite(px0) and math.isfinite(px1)) or px0 <= 0:
        return None
    return (px1 / px0) - 1.0


def _build_dataset(targets: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Merge in forward returns and costs
    unique_tics = sorted(targets["ticker"].unique().tolist())
    px_map = {t: _read_price(t) for t in unique_tics}
    cost_map = _load_cost_bps_map()

    rows = []
    for _, r in targets.iterrows():
        t = r["ticker"]
        d = pd.Timestamp(r["date"])
        px = px_map.get(t)
        fr = _fwd_return(px, d, HORIZON_DAYS)
        if fr is None:
            continue
        cost_bps = cost_map.get(t, DEFAULT_COST_BPS)
        reward = fr - (cost_bps / 10000.0)  # net after cost
        rows.append(
            {
                "date": d,
                "ticker": t,
                "base_w": float(r["base_w"]),
                "final_mult": float(r["final_mult"]),
                "target_w": float(r["target_w"]),
                "reward": float(reward),
            }
        )
        if len(rows) >= MAX_TRAIN_ROWS:
            break

    data = pd.DataFrame(rows)
    # Separate a "current policy" set = last date present (apply to that)
    if data.empty:
        # fall back: zero-weights
        return data, pd.DataFrame(columns=["date", "ticker", "base_w", "final_mult", "target_w"])
    last_day = targets["date"].max()
    current = targets[targets["date"] == last_day].copy()
    current = current[["date", "ticker", "base_w", "final_mult", "target_w"]]
    return data, current


def _fit_ridge(X: np.ndarray, y: np.ndarray, l2: float) -> np.ndarray:
    # Closed-form ridge: (X'X + λI)^(-1) X' y
    n_features = X.shape[1]
    I = np.eye(n_features)
    A = X.T @ X + l2 * I
    b = X.T @ y
    try:
        w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        w = np.linalg.pinv(A) @ b
    return w


def _standardize(train: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, dict]:
    mu = {}
    sig = {}
    t = train.copy()
    for c in cols:
        v = pd.to_numeric(t[c], errors="coerce")
        m = float(v.mean()) if len(v) else 0.0
        s = float(v.std(ddof=0)) if len(v) else 1.0
        if s == 0:
            s = 1.0
        mu[c], sig[c] = m, s
        t[c] = (v - m) / s
    return t, {"mu": mu, "sig": sig}


def _apply_standardize(df: pd.DataFrame, cols: list[str], stats: dict) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        m = stats["mu"].get(c, 0.0)
        s = stats["sig"].get(c, 1.0)
        out[c] = (pd.to_numeric(out[c], errors="coerce") - m) / (s if s != 0 else 1.0)
    return out


def _sigmoid(x):
    # numerically stable-ish
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)

    targets = _load_targets()
    train, current = _build_dataset(targets)

    features = ["base_w", "final_mult", "target_w"]
    train_std, stats = _standardize(train, features)

    # Design matrix with bias
    X = (
        np.c_[np.ones(len(train_std)), train_std[features].values]
        if not train_std.empty
        else np.zeros((0, 1 + len(features)))
    )
    y = train_std["reward"].values if not train_std.empty else np.zeros((0,))

    # Fit
    if len(train_std) >= 12:  # need some data
        w = _fit_ridge(X, y, RIDGE_L2)
    else:
        # Not enough data → fallback tiny weights
        w = np.array([0.0] * (1 + len(features)), dtype=float)

    # Apply to most recent date (current)
    curr_std = _apply_standardize(current, features, stats) if not current.empty else current
    Xc = (
        np.c_[np.ones(len(curr_std)), curr_std[features].values]
        if not curr_std.empty
        else np.zeros((0, 1 + len(features)))
    )
    score = (Xc @ w) if len(curr_std) else np.array([])

    # Size factor & w_rl with caps
    size_factor = _sigmoid(SIGMOID_GAIN * score) if len(score) else np.array([])
    # Kelly-style cap on top of per-name cap
    kelly_cap = KELLY_FRACTION * PER_NAME_CAP
    w_rl = (
        np.minimum(
            current["target_w"].values * size_factor * (1.0 + KELLY_FRACTION),
            PER_NAME_CAP,
        )
        if len(curr_std)
        else np.array([])
    )

    out = pd.DataFrame(
        {
            "date": current["date"].values if len(curr_std) else [],
            "ticker": current["ticker"].values if len(curr_std) else [],
            "target_w": current["target_w"].values if len(curr_std) else [],
            "size_factor": np.round(size_factor, 6) if len(curr_std) else [],
            "w_rl": np.round(w_rl, 6) if len(curr_std) else [],
            "notes": [f"kelly_cap~{kelly_cap:.3f}; per_name_cap={PER_NAME_CAP:.2%}"]
            * (len(curr_std) if len(curr_std) else 0),
        }
    )

    out.to_csv(OUT_CSV, index=False)

    diag = {
        "as_of": (None if targets.empty else str(pd.to_datetime(targets["date"].max()).date())),
        "universe": int(targets["ticker"].nunique()),
        "train_rows": int(len(train)),
        "current_rows": int(len(current)),
        "ridge_l2": RIDGE_L2,
        "sigmoid_gain": SIGMOID_GAIN,
        "kelly_fraction": KELLY_FRACTION,
        "per_name_cap": PER_NAME_CAP,
        "default_cost_bps": DEFAULT_COST_BPS,
        "horizon_days": HORIZON_DAYS,
        "coef": {
            "bias": float(w[0]) if len(w) else 0.0,
            "base_w": float(w[1]) if len(w) >= 2 else 0.0,
            "final_mult": float(w[2]) if len(w) >= 3 else 0.0,
            "target_w": float(w[3]) if len(w) >= 4 else 0.0,
        },
        "standardize": stats,
        "files": {
            "targets_csv": str(TARGETS_CSV),
            "fills_csv": str(FILLS_CSV),
            "out_csv": str(OUT_CSV),
        },
        "notes": "Offline-only sizing policy. Reward = fwd-5d return minus cost. We fit ridge on features and produce a capped weight w_rl. No live routing.",
    }
    OUT_JSON.write_text(json.dumps(diag, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "sizing_csv": str(OUT_CSV),
                "diag_json": str(OUT_JSON),
                "rows": int(len(out)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

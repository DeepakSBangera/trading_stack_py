from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ===== Paths =====
ROOT = Path(r"F:\Projects\trading_stack_py")
DATA = ROOT / "data" / "prices"
REPORTS = ROOT / "reports"

# Inputs (from earlier weeks)
TARGETS_CSV = REPORTS / "wk11_blend_targets.csv"  # needs: date,ticker,base_w,target_w (we use target_w)
# Outputs
OUT_CSV = REPORTS / "wk24_black_litterman_compare.csv"
OUT_JSON = REPORTS / "w24_bl_summary.json"

# Knobs
HIST_DAYS_RET = 252  # lookback for returns/cov
MOM_DAYS = 63  # momentum window to form a simple view
TAU = 0.05  # BL tau (uncertainty on prior)
RISK_AVERSION = 3.0  # lambda; effective when mapping μ -> weights
SHRINK_TO_DIAG = 0.15  # simple shrinkage intensity toward diagonal
CLIP_LONG_ONLY = True  # set negatives to zero for long-only portfolio


# ---------- helpers ----------
def _load_targets_lastday(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    last_day = df["date"].max()
    df = df[df["date"] == last_day].copy()
    # prefer target_w if exists; else fall back to base_w or equal-weight
    wcol = "target_w" if "target_w" in df.columns else ("base_w" if "base_w" in df.columns else None)
    if wcol is None:
        # create equal weight
        df["target_w"] = 1.0 / len(df)
        wcol = "target_w"
    df["w_mv"] = pd.to_numeric(df[wcol], errors="coerce").fillna(0.0)
    df["ticker"] = df["ticker"].astype(str)
    df = df[["ticker", "w_mv"]]
    # normalize
    s = df["w_mv"].clip(lower=0).sum()
    df["w_mv"] = (df["w_mv"].clip(lower=0) / s) if s > 0 else (1.0 / len(df))
    return df


def _load_price_panel(tickers: list[str]) -> pd.DataFrame:
    frames = []
    for t in tickers:
        p = DATA / f"{t}.parquet"
        if not p.exists():
            continue
        try:
            x = pd.read_parquet(p)
            # find date & close columns robustly
            cols = {c.lower(): c for c in x.columns}
            dcol = cols.get("date") or cols.get("dt")
            ccol = cols.get("close") or cols.get("px_close") or cols.get("price")
            if not dcol or not ccol:
                continue
            x[dcol] = pd.to_datetime(x[dcol], errors="coerce")
            x = x[[dcol, ccol]].dropna()
            x = x.rename(columns={dcol: "date", ccol: t}).set_index("date").sort_index()
            frames.append(x)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    panel = pd.concat(frames, axis=1, join="inner").sort_index()
    return panel


def _pct_returns(prices: pd.DataFrame) -> pd.DataFrame:
    rets = prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="all")
    return rets


def _shrink_cov(cov: np.ndarray, shrink: float) -> np.ndarray:
    # simple shrink to diagonal (not Ledoit-Wolf, but stable enough)
    d = np.diag(np.diag(cov))
    return (1.0 - shrink) * cov + shrink * d


def _black_litterman(
    prior_mu: np.ndarray, cov: np.ndarray, tickers: list[str], mom_signal: pd.Series
) -> tuple[np.ndarray, np.ndarray, dict]:
    n = len(tickers)
    Σ = cov
    π = prior_mu.reshape(-1, 1)

    # Views: identity with q proportional to z-scored momentum
    # Compute z-scores on mom_signal aligned to tickers
    mom = mom_signal.reindex(tickers).fillna(0.0)
    if mom.std(ddof=0) < 1e-12:
        z = np.zeros(n)
    else:
        z = ((mom - mom.mean()) / (mom.std(ddof=0) + 1e-12)).values

    P = np.eye(n)
    q = z.reshape(-1, 1) * 0.005  # 50 bps tilt for +1 z, scale as needed
    # View uncertainty Ω proportional to diag of Σ
    omega = np.diag(np.maximum(np.diag(Σ) * 0.50, 1e-8))  # fairly uncertain views

    # Posterior (standard BL)
    inv_tauΣ = np.linalg.pinv(TAU * Σ)
    middle = P.T @ np.linalg.pinv(omega) @ P
    Σ_post = np.linalg.pinv(inv_tauΣ + middle)
    μ_post = Σ_post @ (inv_tauΣ @ π + P.T @ np.linalg.pinv(omega) @ q)

    # Map μ to weights (unconstrained MV solution)
    w_uncon = (1.0 / RISK_AVERSION) * (np.linalg.pinv(Σ) @ μ_post).flatten()

    # Normalize to sum 1 (allow negatives first)
    if np.allclose(w_uncon.sum(), 0.0):
        w_uncon = np.ones(n) / n
    else:
        w_uncon = w_uncon / np.sum(w_uncon)

    w_long = w_uncon.copy()
    if CLIP_LONG_ONLY:
        w_long = np.clip(w_long, 0, None)
        s = w_long.sum()
        w_long = w_long / s if s > 0 else np.ones(n) / n

    diag = {
        "tau": TAU,
        "risk_aversion": RISK_AVERSION,
        "view_scale_bps_per_z": 50,
        "omega_from_var_pct": 50,
    }
    return w_uncon, w_long, diag


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)

    # 1) Load targets (MV baseline weights on last trading day)
    base = _load_targets_lastday(TARGETS_CSV)
    tickers = base["ticker"].tolist()

    # 2) Load price panel and compute returns
    panel = _load_price_panel(tickers)
    if panel.empty:
        # Fall back to equal weight if no price data
        base["w_bl_raw"] = 1.0 / len(base)
        base["w_bl_longonly"] = base["w_bl_raw"]
        base["delta_bl_vs_mv"] = base["w_bl_longonly"] - base["w_mv"]
        base.to_csv(OUT_CSV, index=False)
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "as_of": None,
                    "universe": len(tickers),
                    "used_prices": False,
                    "reason": "No price panel available; emitted equal-weight BL.",
                },
                f,
                indent=2,
            )
        print(
            json.dumps(
                {
                    "out_csv": str(OUT_CSV),
                    "rows": int(base.shape[0]),
                    "used_prices": False,
                },
                indent=2,
            )
        )
        return

    # restrict to last HIST_DAYS_RET
    panel = panel.iloc[-HIST_DAYS_RET:].copy()
    as_of = None if panel.empty else str(panel.index.max().date())

    rets = _pct_returns(panel).dropna(how="any", axis=0)
    if rets.empty:
        raise ValueError("No overlapping returns for the requested window.")

    # 3) Prior mean (π) from reverse optimization using MV weights as 'market'
    #    π = λ Σ w_mkt
    μ_hist = rets.mean().values  # fallback if Σ or mapping fails
    Σ_emp = np.cov(rets.values, rowvar=False)
    Σ_emp = _shrink_cov(Σ_emp, SHRINK_TO_DIAG)

    try:
        w_mkt = base.set_index("ticker").reindex(panel.columns)["w_mv"].fillna(0).values.reshape(-1, 1)
        prior_mu = (RISK_AVERSION * (Σ_emp @ w_mkt)).flatten()
    except Exception:
        prior_mu = μ_hist

    # 4) Momentum view (q) from MOM_DAYS total return
    mom_window = min(MOM_DAYS, rets.shape[0])
    mom = (panel.iloc[-1] / panel.iloc[-mom_window] - 1.0).rename("mom")

    # 5) Black–Litterman posterior and weights
    w_bl_raw, w_bl_long, diag = _black_litterman(prior_mu, Σ_emp, list(panel.columns), mom)

    # 6) Assemble output table aligned to input tickers
    out = base.set_index("ticker")[["w_mv"]].copy()
    tmp = pd.DataFrame(
        {
            "ticker": list(panel.columns),
            "w_bl_raw": w_bl_raw,
            "w_bl_longonly": w_bl_long,
        }
    ).set_index("ticker")

    out = out.join(tmp, how="left")
    # For tickers missing from panel, backfill equal weight for BL columns
    missing = out["w_bl_longonly"].isna()
    if missing.any():
        eq = 1.0 / max(1, (~out["w_bl_longonly"].isna()).sum())
        out.loc[missing, "w_bl_raw"] = eq
        out.loc[missing, "w_bl_longonly"] = eq

    out["delta_bl_vs_mv"] = out["w_bl_longonly"] - out["w_mv"]
    out = out.reset_index()

    # 7) Write artifacts
    out.to_csv(OUT_CSV, index=False)
    summary = {
        "as_of": as_of,
        "universe": int(out.shape[0]),
        "hist_days": HIST_DAYS_RET,
        "mom_days": MOM_DAYS,
        "tau": TAU,
        "risk_aversion": RISK_AVERSION,
        "shrink_to_diag": SHRINK_TO_DIAG,
        "used_prices": True,
        "weights_sum_mv": float(out["w_mv"].sum()),
        "weights_sum_bl_longonly": float(out["w_bl_longonly"].sum()),
        "delta_abs_mean": float(out["delta_bl_vs_mv"].abs().mean()),
        "delta_abs_p90": float(np.percentile(out["delta_bl_vs_mv"].abs(), 90)),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        json.dumps(
            {
                "out_csv": str(OUT_CSV),
                "out_json": str(OUT_JSON),
                "rows": int(out.shape[0]),
                "as_of": as_of,
                "delta_abs_mean": round(summary["delta_abs_mean"], 6),
                "delta_abs_p90": round(summary["delta_abs_p90"], 6),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

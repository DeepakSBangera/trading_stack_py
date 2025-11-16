from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"F:\Projects\trading_stack_py")
DATA_PQ = ROOT / "data" / "prices"
DATA_CSV = ROOT / "data" / "csv"
REPORTS = ROOT / "reports"

TARGETS_CSV = REPORTS / "wk11_blend_targets.csv"  # from W11
BENCH_CSV = REPORTS / "benchmarks.csv"  # from W0 (optional)
OUT_DETAIL = REPORTS / "wk23_factor_risk_model.csv"
OUT_WEEKLY = REPORTS / "factor_exposure_weekly.csv"
OUT_SUMMARY = REPORTS / "w23_factor_summary.json"

# factor windows
MOM_WDAYS = 126  # ~6 months
VOL_WDAYS = 21  # 1m realized-vol proxy
BETA_WDAYS = 126  # window for beta estimation


def _git_sha_short() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT),
            stderr=subprocess.DEVNULL,
        )
        return out.decode("utf-8", "ignore").strip()
    except Exception:
        return "????????"


def _load_prices_any(ticker: str) -> pd.DataFrame | None:
    p_parq = DATA_PQ / f"{ticker}.parquet"
    p_csv = DATA_CSV / f"{ticker}.csv"
    try:
        if p_parq.exists():
            df = pd.read_parquet(p_parq)
        elif p_csv.exists():
            df = pd.read_csv(p_csv)
        else:
            return None
    except Exception:
        return None
    if df is None or df.empty:
        return None
    cols = {c.lower(): c for c in df.columns}
    dcol = cols.get("date") or cols.get("dt")
    ccol = cols.get("close") or cols.get("px_close") or cols.get("price")
    if not dcol or not ccol:
        return None
    out = df[[dcol, ccol]].copy()
    out[dcol] = pd.to_datetime(out[dcol], errors="coerce")
    out = out.dropna(subset=[dcol, ccol]).rename(columns={dcol: "date", ccol: "close"})
    out = out.sort_values("date").reset_index(drop=True)
    return out


def _load_benchmark_series() -> pd.DataFrame | None:
    # Try W0 benchmarks.csv (expects columns date,ticker,close) and pick NIFTYBEES if present, else first
    if not BENCH_CSV.exists():
        return None
    try:
        b = pd.read_csv(BENCH_CSV)
    except Exception:
        return None
    cols = {c.lower(): c for c in b.columns}
    dcol = cols.get("date")
    tcol = cols.get("ticker")
    ccol = cols.get("close")
    if not all([dcol, tcol, ccol]):
        return None
    b[dcol] = pd.to_datetime(b[dcol], errors="coerce")
    b = b.dropna(subset=[dcol, tcol, ccol])
    # prefer NIFTYBEES
    choices = ["NIFTYBEES", "NIFTY 50", "NIFTY50"]
    pick = None
    for name in choices:
        sel = b[b[tcol].astype(str).str.upper() == name]
        if not sel.empty:
            pick = sel[[dcol, ccol]].copy()
            break
    if pick is None:
        # fallback to first ticker
        t0 = b[tcol].iloc[0]
        pick = b[b[tcol] == t0][[dcol, ccol]].copy()
    pick = pick.rename(columns={dcol: "date", ccol: "mkt_close"}).sort_values("date").reset_index(drop=True)
    return pick


def _pct_ret(s: pd.Series) -> pd.Series:
    return s.pct_change()


def _zscore(s: pd.Series) -> pd.Series:
    if s.std(ddof=0) == 0 or s.isna().all():
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.mean()) / (s.std(ddof=0) + 1e-12)


def _rolling_beta(y: pd.Series, x: pd.Series, window: int) -> float | None:
    # simple OLS beta using last `window` aligned points
    df = pd.concat([y, x], axis=1).dropna()
    if df.shape[0] < max(20, window // 4):
        return None
    df = df.iloc[-window:]
    xr = df.iloc[:, 1]
    yr = df.iloc[:, 0]
    vx = float(np.var(xr, ddof=0))
    if vx <= 0 or not math.isfinite(vx):
        return None
    cov = float(np.cov(xr, yr, ddof=0)[0, 1])
    beta = cov / vx
    return beta if math.isfinite(beta) else None


def _collect_watchlist() -> tuple[list[str], pd.Timestamp]:
    if not TARGETS_CSV.exists():
        raise FileNotFoundError(f"Missing {TARGETS_CSV}; run W11 first.")
    t = pd.read_csv(TARGETS_CSV)
    cols = {c.lower(): c for c in t.columns}
    dcol = cols.get("date")
    tcol = cols.get("ticker")
    wcol = cols.get("target_w") or cols.get("base_w")
    if not dcol or not tcol:
        raise ValueError("wk11_blend_targets.csv must have columns: date, ticker, target_w")
    t[dcol] = pd.to_datetime(t[dcol], errors="coerce")
    t = t.dropna(subset=[dcol, tcol])
    # pick last date snapshot for weights
    as_of = t[dcol].max()
    snap = t[t[dcol] == as_of].copy()
    if wcol and wcol in snap.columns:
        snap = snap[[tcol, wcol]].rename(columns={tcol: "ticker", wcol: "weight"})
    else:
        snap = snap[[tcol]].rename(columns={tcol: "ticker"})
        snap["weight"] = 1.0 / snap.shape[0]
    snap["weight"] = pd.to_numeric(snap["weight"], errors="coerce").fillna(0.0)
    return sorted(snap["ticker"].astype(str).unique()), as_of


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)

    tickers, as_of = _collect_watchlist()
    bench = _load_benchmark_series()

    # build panel of returns
    panel = {}
    for tic in tickers:
        px = _load_prices_any(tic)
        if px is None or px.empty:
            continue
        panel[tic] = px

    # find common last date across panel
    last_dates = [df["date"].max() for df in panel.values() if not df.empty]
    if last_dates:
        last_common = min(last_dates)  # conservative
    else:
        last_common = as_of

    # build daily pct returns aligned
    all_rets = []
    for tic, df in panel.items():
        df = df[df["date"] <= last_common].copy()
        df["ret"] = _pct_ret(df["close"])
        all_rets.append(df[["date", "ret"]].assign(ticker=tic))
    if not all_rets:
        # nothing to do
        detail = pd.DataFrame(
            columns=[
                "ticker",
                "mom_126",
                "vol_21",
                "beta_126",
                "mom_z",
                "vol_z",
                "beta_z",
                "as_of",
            ]
        )
        detail.to_csv(OUT_DETAIL, index=False)
        with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "as_of": str(pd.Timestamp(as_of).date()),
                    "tickers": 0,
                    "note": "no price data found",
                    "git_sha8": _git_sha_short(),
                },
                f,
                indent=2,
            )
        print(json.dumps({"out_detail_csv": str(OUT_DETAIL), "tickers": 0}, indent=2))
        return

    rets = pd.concat(all_rets, ignore_index=True)
    rets = rets.dropna(subset=["date"]).sort_values(["ticker", "date"]).reset_index(drop=True)

    # compute per-ticker features
    rows = []
    # prepare benchmark returns if available
    if bench is not None and not bench.empty:
        bm = bench[bench["date"] <= last_common].copy()
        bm["mkt_ret"] = _pct_ret(bm["mkt_close"])
        bm = bm[["date", "mkt_ret"]]
    else:
        bm = None

    for tic in tickers:
        r = rets[rets["ticker"] == tic].copy()
        if r.empty:
            rows.append({"ticker": tic, "mom_126": np.nan, "vol_21": np.nan, "beta_126": np.nan})
            continue
        r = r.sort_values("date").reset_index(drop=True)
        # MOM: cumulative total return over 126d (excluding most recent 1d to avoid micro reversal)
        # safe window slice
        tail = r.iloc[-(MOM_WDAYS + 1) :].copy() if r.shape[0] >= (MOM_WDAYS + 1) else r.copy()
        mom_126 = float((1.0 + tail["ret"].iloc[:-1].fillna(0.0)).prod() - 1.0) if tail.shape[0] >= 10 else np.nan

        # VOL: realized stdev over 21d
        vtail = r.iloc[-VOL_WDAYS:].copy() if r.shape[0] >= VOL_WDAYS else r.copy()
        vol_21 = float(vtail["ret"].std(ddof=0)) if vtail.shape[0] >= 5 else np.nan

        # BETA: OLS beta vs benchmark; if not available, beta vs equal-weight market of peers
        beta_126 = np.nan
        if bm is not None and not bm.empty:
            m = pd.merge(r[["date", "ret"]], bm, on="date", how="inner")
            if not m.empty:
                beta_126 = _rolling_beta(m["ret"], m["mkt_ret"], window=BETA_WDAYS)
        if bm is None or (not math.isfinite(beta_126) if isinstance(beta_126, float) else True):
            # fallback synthetic market: mean of peer returns each day (exclude self)
            peers = rets[rets["ticker"] != tic]
            mkt = peers.groupby("date", as_index=False)["ret"].mean().rename(columns={"ret": "mkt_ret"})
            m2 = pd.merge(r[["date", "ret"]], mkt, on="date", how="inner")
            if not m2.empty:
                beta_126 = _rolling_beta(m2["ret"], m2["mkt_ret"], window=BETA_WDAYS)
        rows.append({"ticker": tic, "mom_126": mom_126, "vol_21": vol_21, "beta_126": beta_126})

    detail = pd.DataFrame(rows)
    # derive z-scores across current universe (handle NaNs)
    for col in ["mom_126", "vol_21", "beta_126"]:
        if col not in detail.columns:
            detail[col] = np.nan

    detail["mom_z"] = _zscore(detail["mom_126"].fillna(detail["mom_126"].median()))
    # lower vol is "good" → invert sign so higher = better risk quality
    detail["vol_inv"] = -detail["vol_21"]
    detail["vol_z"] = _zscore(detail["vol_inv"].fillna(detail["vol_inv"].median()))
    detail["beta_z"] = _zscore(detail["beta_126"].fillna(detail["beta_126"].median()))
    detail["as_of"] = pd.Timestamp(last_common).date()

    # write detail
    REPORTS.mkdir(parents=True, exist_ok=True)
    keep_cols = [
        "ticker",
        "mom_126",
        "vol_21",
        "beta_126",
        "mom_z",
        "vol_z",
        "beta_z",
        "as_of",
    ]
    detail[keep_cols].to_csv(OUT_DETAIL, index=False)

    # portfolio exposure = weight * zscore
    t = pd.read_csv(TARGETS_CSV)
    t["date"] = pd.to_datetime(t["date"], errors="coerce")
    snap = t[t["date"] == t["date"].max()].copy()
    wcol = "target_w" if "target_w" in snap.columns else ("base_w" if "base_w" in snap.columns else None)
    if wcol is None:
        snap[wcol] = 1.0 / max(1, len(tickers))
    snap = snap.rename(columns={"ticker": "ticker"}).merge(
        detail[["ticker", "mom_z", "vol_z", "beta_z"]], on="ticker", how="left"
    )
    snap["weight"] = pd.to_numeric(snap.get(wcol, 0.0), errors="coerce").fillna(0.0)

    port_mom = float((snap["weight"] * snap["mom_z"]).sum())
    port_vol = float((snap["weight"] * snap["vol_z"]).sum())
    port_beta = float((snap["weight"] * snap["beta_z"]).sum())

    # append/update weekly exposure time series
    wk_row = pd.DataFrame(
        [
            {
                "date": pd.Timestamp(last_common).date(),
                "mom_z_exp": port_mom,
                "vol_z_exp": port_vol,
                "beta_z_exp": port_beta,
            }
        ]
    )
    if OUT_WEEKLY.exists():
        try:
            prev = pd.read_csv(OUT_WEEKLY)
            prev["date"] = pd.to_datetime(prev["date"], errors="coerce").dt.date
            outw = pd.concat([prev, wk_row], ignore_index=True).drop_duplicates(subset=["date"], keep="last")
        except Exception:
            outw = wk_row.copy()
    else:
        outw = wk_row.copy()
    outw.to_csv(OUT_WEEKLY, index=False)

    summary = {
        "as_of": str(pd.Timestamp(last_common).date()),
        "universe": len(tickers),
        "detail_csv": str(OUT_DETAIL),
        "exposure_weekly_csv": str(OUT_WEEKLY),
        "portfolio_exposures_z": {
            "momentum": round(port_mom, 4),
            "vol_inv": round(port_vol, 4),
            "beta": round(port_beta, 4),
        },
        "notes": "z-score exposures of current wk11 weights; vol_inv means lower vol → higher score",
        "git_sha8": _git_sha_short(),
    }
    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        json.dumps(
            {
                "out_detail_csv": str(OUT_DETAIL),
                "out_weekly_csv": str(OUT_WEEKLY),
                "portfolio_z_exposures": summary["portfolio_exposures_z"],
                "universe": len(tickers),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

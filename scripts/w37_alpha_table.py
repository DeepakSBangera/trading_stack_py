from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

"""
W37 — Alpha Table & Promotion Ladder

What this does (robust, self-contained):
1) Loads target universe/dates from reports/wk11_blend_targets.csv (date,ticker,target_w).
2) Loads price panels from data/prices/<ticker>.parquet (needs date + close).
3) Derives a small set of canonical signals from price history so it runs even if no signal file exists:
   - mom_252, mom_126, mom_21  (price momentum)
   - vol_inv_63                 (inverse volatility)
   - mom_21_voladj             (21d momentum scaled by 63d vol)
4) Computes forward returns at 5d and 21d horizons (configurable).
5) Cross-sectional IC (Spearman) per day per signal vs forward return; then roll stats.
6) Half-life from AR(1) of IC series (days).
7) Reads optional W5 DSR file: reports/wk5_walkforward_dsr.csv (cols: signal, dsr, pbo?), merges if present.
8) Emits:
   - reports/ic_timeseries.csv (tidy: date, signal, horizon, ic)
   - reports/wk37_alpha_table.csv (summary: mean_ic_21/63, t-stat, half-life, coverage, DSR/PBO if available, action)
   - reports/w37_alpha_table_summary.json (top promote/retire lists)

Promotion ladder (default policy):
- Promote if mean_ic_63 >= 0.02 AND half_life_days >= 20 AND coverage >= 100 days AND (if DSR present) dsr > 0.
- Retire if mean_ic_63 <= 0.0 OR half_life_days < 10 OR coverage < 60.
- Else Keep.

You can tune thresholds at the top.
"""

ROOT = Path(r"F:\Projects\trading_stack_py")
REPORTS = ROOT / "reports"
DATA = ROOT / "data" / "prices"

TARGETS_CSV = REPORTS / "wk11_blend_targets.csv"
IC_TIMESERIES = REPORTS / "ic_timeseries.csv"  # (overwritten)
ALPHA_TABLE_CSV = REPORTS / "wk37_alpha_table.csv"
SUMMARY_JSON = REPORTS / "w37_alpha_table_summary.json"

# Optional DSR file from W5
W5_DSR_CSV_OPT = REPORTS / "wk5_walkforward_dsr.csv"

# --- knobs ---
FWD_HORIZONS = [5, 21]  # trading days
ROLL_WINDOWS = [21, 63]  # for mean IC summaries
PROMOTE_MIN_MEAN_IC_63 = 0.02
PROMOTE_MIN_HALFLIFE = 20
PROMOTE_MIN_COVERAGE = 100
RETIRE_MAX_MEAN_IC_63 = 0.00
RETIRE_MAX_HALFLIFE = 10
RETIRE_MIN_COVERAGE = 60


# -------- utils --------
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


def _load_panel(tickers: List[str]) -> pd.DataFrame:
    frames = []
    for t in sorted(set(tickers)):
        px = _read_prices_for(t)
        if px is None or px.empty:
            continue
        frames.append(px.set_index("date").rename(columns={"close": t})[[t]])
    if not frames:
        return pd.DataFrame()
    panel = pd.concat(frames, axis=1, join="outer").sort_index()
    return panel


def _signals_from_prices(panel: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Derive simple signals that exist for any liquid series.
    Returns dict of wide DataFrames (index=date, columns=tickers).
    """
    rets = panel.pct_change()

    # momentum as cumulative return over window
    def mom(win: int) -> pd.DataFrame:
        return panel / panel.shift(win) - 1.0

    vol_63 = rets.rolling(63).std()
    vol_inv_63 = 1.0 / vol_63.replace(0, np.nan)

    sigs = {
        "mom_252": mom(252),
        "mom_126": mom(126),
        "mom_21": mom(21),
        "vol_inv_63": vol_inv_63,
        "mom_21_voladj": mom(21) / vol_63.replace(0, np.nan),
    }
    # Winsorize light to reduce extreme ranks effect
    for k, df in sigs.items():
        sigs[k] = df.clip(lower=df.quantile(0.01), upper=df.quantile(0.99), axis=1)
    return sigs


def _forward_returns(panel: pd.DataFrame, horizon: int) -> pd.DataFrame:
    return panel.shift(-horizon) / panel - 1.0


def _spearman_ic(sig: pd.Series, fwd: pd.Series) -> float:
    # require at least 5 names to be meaningful
    s = pd.concat([sig.rename("s"), fwd.rename("r")], axis=1).dropna()
    if s.shape[0] < 5:
        return np.nan
    return s["s"].rank().corr(s["r"].rank(), method="spearman")


def _ic_timeseries_for(signal_wide: pd.DataFrame, fwd_wide: pd.DataFrame) -> pd.Series:
    # Align
    common_idx = signal_wide.index.intersection(fwd_wide.index)
    common_cols = signal_wide.columns.intersection(fwd_wide.columns)
    S = signal_wide.reindex(index=common_idx, columns=common_cols)
    R = fwd_wide.reindex(index=common_idx, columns=common_cols)
    out = []
    for dt, row in S.iterrows():
        ic = _spearman_ic(row, R.loc[dt])
        out.append((dt, ic))
    return pd.Series({d: v for d, v in out}).sort_index()


def _half_life_days(series: pd.Series) -> float:
    s = series.dropna()
    if s.size < 10:
        return 0.0
    # AR(1) via autocorr at lag 1
    phi = float(s.autocorr(lag=1))
    if not np.isfinite(phi) or phi <= 0.0 or phi >= 0.9999:
        return 0.0 if phi <= 0 else 1e6  # if near 1, extremely long; clamp
    # HL = ln(0.5) / ln(phi)
    try:
        hl = math.log(0.5) / math.log(phi)
    except Exception:
        hl = 0.0
    # series is daily IC; treat as trading-day half-life
    return float(max(0.0, hl))


def _tstat(x: pd.Series) -> float:
    s = x.dropna()
    if s.size < 5:
        return np.nan
    m = s.mean()
    sd = s.std(ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return np.nan
    return float(m / (sd / math.sqrt(s.size)))


def _read_targets() -> pd.DataFrame:
    if not TARGETS_CSV.exists():
        raise FileNotFoundError(f"Missing {TARGETS_CSV}; run W11 first.")
    df = pd.read_csv(TARGETS_CSV)
    cols = {c.lower(): c for c in df.columns}
    if "date" not in cols or "ticker" not in cols:
        raise ValueError("wk11_blend_targets.csv must contain date,ticker,target_w")
    df = df.rename(columns={cols["date"]: "date", cols["ticker"]: "ticker"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["ticker"] = df["ticker"].astype(str)
    return df


def _read_dsr_opt() -> pd.DataFrame | None:
    if not W5_DSR_CSV_OPT.exists():
        return None
    try:
        d = pd.read_csv(W5_DSR_CSV_OPT)
        # expected cols: signal, dsr, pbo (best-effort)
        cols = {c.lower(): c for c in d.columns}
        s = cols.get("signal")
        dsr = cols.get("dsr")
        pbo = cols.get("pbo")
        if not s or not dsr:
            return None
        out = d.rename(columns={s: "signal"})
        if dsr:
            out = out.rename(columns={dsr: "dsr"})
        if pbo:
            out = out.rename(columns={pbo: "pbo"})
        out["signal"] = out["signal"].astype(str)
        return out[["signal"] + [c for c in ["dsr", "pbo"] if c in out.columns]]
    except Exception:
        return None


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    t = _read_targets()
    tickers = sorted(t["ticker"].unique())
    panel = _load_panel(tickers)
    if panel.empty:
        # Produce empty but valid outputs
        pd.DataFrame(columns=["date", "signal", "horizon", "ic"]).to_csv(
            IC_TIMESERIES, index=False
        )
        pd.DataFrame(
            columns=[
                "signal",
                "coverage_days",
                "mean_ic_21",
                "mean_ic_63",
                "tstat_63",
                "half_life_days",
                "action",
                "dsr",
                "pbo",
            ]
        ).to_csv(ALPHA_TABLE_CSV, index=False)
        SUMMARY_JSON.write_text(
            json.dumps({"rows": 0, "promote": [], "retire": []}, indent=2),
            encoding="utf-8",
        )
        print(json.dumps({"alpha_table": str(ALPHA_TABLE_CSV), "rows": 0}, indent=2))
        return

    # 1) Derive signals
    sigs = _signals_from_prices(panel)

    # 2) Forward returns
    fwd_map = {h: _forward_returns(panel, h) for h in FWD_HORIZONS}

    # 3) Daily IC series per signal & horizon
    ts_rows = []
    for sig_name, sig_wide in sigs.items():
        for h, fwd in fwd_map.items():
            ic_s = _ic_timeseries_for(sig_wide, fwd)
            if not ic_s.empty:
                ts_rows.append(
                    pd.DataFrame(
                        {
                            "date": ic_s.index,
                            "signal": sig_name,
                            "horizon": h,
                            "ic": ic_s.values,
                        }
                    )
                )
    if ts_rows:
        ic_ts = pd.concat(ts_rows, axis=0).sort_values(["signal", "horizon", "date"])
    else:
        ic_ts = pd.DataFrame(columns=["date", "signal", "horizon", "ic"])
    ic_ts.to_csv(IC_TIMESERIES, index=False)

    # 4) Build alpha table summary (use horizon=21 for half-life; compute mean over 21/63-day windows)
    summaries = []
    dsr_df = _read_dsr_opt()

    for sig_name in sigs.keys():
        ic_21d = ic_ts[(ic_ts["signal"] == sig_name) & (ic_ts["horizon"] == 21)].copy()
        ic_5d = ic_ts[(ic_ts["signal"] == sig_name) & (ic_ts["horizon"] == 5)].copy()

        # coverage days = count of non-na IC
        coverage = int(ic_21d["ic"].notna().sum())
        # rolling means (21d & 63d) and t-stat on 63d window
        mean_21 = float(ic_21d["ic"].tail(21).mean()) if coverage > 0 else np.nan
        last_63 = ic_21d["ic"].tail(63)
        mean_63 = float(last_63.mean()) if last_63.size > 0 else np.nan
        tstat_63 = _tstat(last_63)

        # half-life from entire 21d-horizon IC series
        hl = _half_life_days(ic_21d.set_index("date")["ic"]) if coverage > 0 else 0.0

        # Promotion ladder
        dsr = np.nan
        pbo = np.nan
        if dsr_df is not None and not dsr_df.empty:
            m = dsr_df[dsr_df["signal"].str.lower() == sig_name.lower()]
            if not m.empty:
                if "dsr" in m.columns:
                    dsr = float(pd.to_numeric(m.iloc[0]["dsr"], errors="coerce"))
                if "pbo" in m.columns:
                    pbo = float(pd.to_numeric(m.iloc[0]["pbo"], errors="coerce"))

        action = "Keep"
        if (
            mean_63 is not None
            and not np.isnan(mean_63)
            and mean_63 >= PROMOTE_MIN_MEAN_IC_63
            and hl >= PROMOTE_MIN_HALFLIFE
            and coverage >= PROMOTE_MIN_COVERAGE
            and (np.isnan(dsr) or dsr > 0)
        ):
            action = "Promote"
        if (
            mean_63 is not None
            and not np.isnan(mean_63)
            and (
                mean_63 <= RETIRE_MAX_MEAN_IC_63
                or hl < RETIRE_MAX_HALFLIFE
                or coverage < RETIRE_MIN_COVERAGE
            )
        ):
            action = "Retire"

        summaries.append(
            {
                "signal": sig_name,
                "coverage_days": coverage,
                "mean_ic_21": round(mean_21 if np.isfinite(mean_21) else np.nan, 5),
                "mean_ic_63": round(mean_63 if np.isfinite(mean_63) else np.nan, 5),
                "tstat_63": round(tstat_63 if np.isfinite(tstat_63) else np.nan, 3),
                "half_life_days": round(float(hl), 2),
                "action": action,
                "dsr": (None if np.isnan(dsr) else round(dsr, 3)),
                "pbo": (None if np.isnan(pbo) else round(pbo, 3)),
            }
        )

    alpha_table = pd.DataFrame(summaries).sort_values(
        ["action", "mean_ic_63"], ascending=[True, False]
    )
    alpha_table.to_csv(ALPHA_TABLE_CSV, index=False)

    promote = alpha_table[alpha_table["action"] == "Promote"]["signal"].tolist()
    retire = alpha_table[alpha_table["action"] == "Retire"]["signal"].tolist()

    summary = {
        "rows": int(alpha_table.shape[0]),
        "promote": promote,
        "retire": retire,
        "files": {
            "alpha_table_csv": str(ALPHA_TABLE_CSV),
            "ic_timeseries_csv": str(IC_TIMESERIES),
        },
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {"alpha_table": str(ALPHA_TABLE_CSV), "promote": promote, "retire": retire},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

# -------- paths --------
ROOT = Path(r"F:\Projects\trading_stack_py")
DATA = ROOT / "data" / "prices"
REPORTS = ROOT / "reports"
OUT_CSV = REPORTS / "wk21_purgedcv_pbo.csv"
OUT_JSON = REPORTS / "w21_summary.json"
TARGETS_CSV = REPORTS / "wk11_blend_targets.csv"  # used to build a daily portfolio return

# -------- knobs --------
TRADING_DAYS_PER_YEAR = 252
K_FOLDS = 5
EMBARGO_DAYS = 3  # small embargo (purging nearby look-ahead leakage)
MIN_VALID_ROWS = 30  # require at least this many days to proceed


# ===== utils =====
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


def _annualize_sharpe(r: pd.Series) -> float:
    r = pd.to_numeric(pd.Series(r).dropna(), errors="coerce").dropna()
    if r.empty:
        return 0.0
    mu = r.mean()
    sd = r.std(ddof=1)
    if not math.isfinite(mu) or not math.isfinite(sd) or sd == 0:
        return 0.0
    return float((mu / sd) * math.sqrt(TRADING_DAYS_PER_YEAR))


def _time_splits(dates: np.ndarray, k_folds: int, embargo_days: int):
    """
    Simple chronological K-fold with embargo around test windows.
    Yields (train_index, test_index) as numpy arrays of row indices.
    """
    n = len(dates)
    fold_sizes = np.full(k_folds, n // k_folds, dtype=int)
    fold_sizes[: n % k_folds] += 1
    starts = np.cumsum(np.concatenate(([0], fold_sizes[:-1])))
    ends = starts + fold_sizes

    for s, e in zip(starts, ends, strict=False):
        test_idx = np.arange(s, e, dtype=int)
        # embargo: remove a few days around test window from train (both sides)
        left_cut = max(0, s - embargo_days)
        right_cut = min(n, e + embargo_days)
        train_idx = np.concatenate([np.arange(0, left_cut, dtype=int), np.arange(right_cut, n, dtype=int)])
        yield train_idx, test_idx


def _safe_to_datetime(col: pd.Series) -> pd.Series:
    out = pd.to_datetime(col, errors="coerce", utc=False)
    return out


# ===== data layer =====
def _load_close_for_ticker(ticker: str) -> pd.DataFrame | None:
    """Load EOD close for a ticker from data/prices/<ticker>.parquet or .csv fallback."""
    p_parq = DATA / f"{ticker}.parquet"
    p_csv = DATA / "csv" / f"{ticker}.csv"
    if p_parq.exists():
        try:
            df = pd.read_parquet(p_parq)
        except Exception:
            df = None
    elif p_csv.exists():
        try:
            df = pd.read_csv(p_csv)
        except Exception:
            df = None
    else:
        return None
    if df is None or df.empty:
        return None

    cols = {c.lower(): c for c in df.columns}
    dcol = cols.get("date") or cols.get("dt") or "date"
    ccol = cols.get("close") or cols.get("px_close") or cols.get("price") or "close"

    if dcol not in df.columns or ccol not in df.columns:
        return None

    df = df[[dcol, ccol]].copy()
    df[dcol] = _safe_to_datetime(df[dcol])
    df = df.dropna(subset=[dcol, ccol]).sort_values(dcol)
    df = df.rename(columns={dcol: "date", ccol: "close"})
    return df


def _build_portfolio_daily_returns(targets_csv: Path) -> pd.DataFrame:
    """
    Build a daily portfolio return series using wk11_blend_targets.csv:
    - Uses 'target_w' per date/ticker as static weights for next-day close-to-close returns.
    - If some tickers lack prices, those rows are skipped.
    Output columns: ['date', 'ret_port']
    """
    if not targets_csv.exists():
        raise FileNotFoundError(f"Missing {targets_csv} — run W11 first.")

    tdf = pd.read_csv(targets_csv)
    # expected cols: date, ticker, target_w (others are ignored)
    cols = {c.lower(): c for c in tdf.columns}
    dcol = cols.get("date")
    ticol = cols.get("ticker")
    wcol = cols.get("target_w") or cols.get("weight") or cols.get("w")

    if not dcol or not ticol or not wcol:
        raise ValueError("wk11_blend_targets.csv must have columns: date, ticker, target_w")

    tdf = tdf[[dcol, ticol, wcol]].copy()
    tdf[dcol] = _safe_to_datetime(tdf[dcol])
    tdf = tdf.dropna(subset=[dcol, ticol, wcol])
    tdf = tdf.rename(columns={dcol: "date", ticol: "ticker", wcol: "weight"}).sort_values(["date", "ticker"])

    # Normalize weights per day (safety)
    tdf["weight"] = pd.to_numeric(tdf["weight"], errors="coerce").fillna(0.0)
    tdf["weight"] = tdf["weight"].clip(lower=-1e9, upper=1e9)
    tdf["date"] = pd.to_datetime(tdf["date"], errors="coerce")
    tdf = tdf.dropna(subset=["date"])

    # Get unique tickers and load closes
    tickers = sorted(tdf["ticker"].astype(str).unique())
    close_map: dict[str, pd.DataFrame] = {}
    for tic in tickers:
        dfc = _load_close_for_ticker(tic)
        if dfc is not None and not dfc.empty:
            close_map[tic] = dfc

    if not close_map:
        # Nothing available → empty series
        return pd.DataFrame(columns=["date", "ret_port"])

    # Build daily return per ticker
    rets = []
    for tic, dfc in close_map.items():
        dfc = dfc.copy()
        dfc["ret"] = dfc["close"].pct_change()
        rets.append(dfc[["date", "ret"]].assign(ticker=tic))

    rdf = pd.concat(rets, ignore_index=True)
    rdf = rdf.dropna(subset=["date", "ret"])

    # Join weights → portfolio return per day = sum_ticker(weight(date,tic) * ret(date,tic))
    j = pd.merge(
        tdf,
        rdf,
        on=["date", "ticker"],
        how="inner",
        validate="many_to_one",
    )
    if j.empty:
        return pd.DataFrame(columns=["date", "ret_port"])

    j["w_ret"] = j["weight"] * j["ret"]
    port = j.groupby("date", as_index=False)["w_ret"].sum().rename(columns={"w_ret": "ret_port"})
    port = port.dropna(subset=["ret_port"]).sort_values("date")
    return port


# ===== evaluation =====
def _oos_eval(daily: pd.DataFrame, k_folds: int, embargo_days: int):
    """
    Returns:
      folds_df: per-fold metrics DataFrame
      summary:  dict with sr_is, sr_oos, t_is, t_oos
    """
    if daily.index.name == "date" and "date" in daily.columns:
        daily = daily.copy()
        daily.index.name = None

    if "date" not in daily.columns:
        daily = daily.reset_index().rename(columns={"index": "date"})

    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
    daily = daily.dropna(subset=["date", "ret_port"]).sort_values("date").reset_index(drop=True)

    if len(daily) < MIN_VALID_ROWS:
        folds_df = pd.DataFrame(columns=["fold", "n_is", "n_oos", "sr_is", "sr_oos"])
        return folds_df, {"sr_is": 0.0, "sr_oos": 0.0, "t_is": 0, "t_oos": 0}

    dates = daily["date"].values
    rets = daily["ret_port"].astype(float).values

    rows = []
    is_series, oos_series = [], []
    t_is = t_oos = 0

    for k, (tr_idx, te_idx) in enumerate(_time_splits(dates, k_folds, embargo_days), start=1):
        r_is = pd.Series(rets[tr_idx])
        r_oos = pd.Series(rets[te_idx])

        sr_is = _annualize_sharpe(r_is)
        sr_oos = _annualize_sharpe(r_oos)

        rows.append(
            {
                "fold": k,
                "n_is": int(r_is.shape[0]),
                "n_oos": int(r_oos.shape[0]),
                "sr_is": round(sr_is, 4),
                "sr_oos": round(sr_oos, 4),
            }
        )
        if not r_is.empty:
            is_series.append(r_is)
            t_is += len(r_is)
        if not r_oos.empty:
            oos_series.append(r_oos)
            t_oos += len(r_oos)

    sr_is_all = _annualize_sharpe(pd.concat(is_series)) if is_series else 0.0
    sr_oos_all = _annualize_sharpe(pd.concat(oos_series)) if oos_series else 0.0

    folds_df = pd.DataFrame(rows)
    summary = {
        "sr_is": round(float(sr_is_all), 4),
        "sr_oos": round(float(sr_oos_all), 4),
        "t_is": int(t_is),
        "t_oos": int(t_oos),
    }
    return folds_df, summary


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)

    # 1) Build daily portfolio returns from W11 targets + prices
    daily = _build_portfolio_daily_returns(TARGETS_CSV)  # columns: date, ret_port

    # 2) K-fold purged CV with embargo
    folds_df, summary = _oos_eval(daily, K_FOLDS, EMBARGO_DAYS)

    # 3) Persist
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    folds_df.to_csv(OUT_CSV, index=False)

    out = {
        "out_csv": str(OUT_CSV),
        "rows": int(folds_df.shape[0]),
        "sr_is": summary["sr_is"],
        "sr_oos": summary["sr_oos"],
        "t_is": summary["t_is"],
        "t_oos": summary["t_oos"],
        "k_folds": K_FOLDS,
        "embargo_days": EMBARGO_DAYS,
        "git_sha8": _git_sha_short(),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(r"F:\Projects\trading_stack_py")
DATA = ROOT / "data" / "prices"
DATA_CSV = ROOT / "data" / "csv"
REPORTS = ROOT / "reports"

TARGETS_CSV = REPORTS / "wk11_blend_targets.csv"
OUT_SUMMARY = REPORTS / "wk22_pit_integrity.csv"
OUT_GAPS = REPORTS / "wk22_pit_gaps.csv"
OUT_JSON = REPORTS / "w22_summary.json"

TRADING_DAYS_PER_WEEK = 5


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


def _safe_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def _load_close_any(ticker: str) -> pd.DataFrame | None:
    p_parq = DATA / f"{ticker}.parquet"
    p_csv = DATA_CSV / f"{ticker}.csv"
    df = None
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
    out[dcol] = _safe_dt(out[dcol])
    out = out.dropna(subset=[dcol, ccol]).rename(columns={dcol: "date", ccol: "close"})
    out = out.sort_values("date").reset_index(drop=True)
    return out


def _business_day_gaps(dates: pd.Series) -> list[pd.Timestamp]:
    """Return missing business dates between first and last, relative to the series’ existing dates."""
    if dates.empty:
        return []
    cal = pd.bdate_range(dates.min(), dates.max(), freq="C")
    present = pd.Series(True, index=pd.to_datetime(dates.values))
    # Missing are business days not present in the index
    missing = [d for d in cal if d not in present.index]
    return missing


def _collect_watchlist() -> tuple[list[str], pd.Timestamp, pd.Timestamp]:
    """Tickers + date window taken from wk11_blend_targets.csv."""
    if not TARGETS_CSV.exists():
        raise FileNotFoundError(f"Missing {TARGETS_CSV}; run W11 first.")
    t = pd.read_csv(TARGETS_CSV)
    cols = {c.lower(): c for c in t.columns}
    dcol = cols.get("date")
    tcol = cols.get("ticker")
    if not dcol or not tcol:
        raise ValueError("wk11_blend_targets.csv must have columns: date, ticker")
    t[dcol] = _safe_dt(t[dcol])
    t = t.dropna(subset=[dcol, tcol])
    tickers = sorted(t[tcol].astype(str).unique())
    dmin, dmax = t[dcol].min(), t[dcol].max()
    return tickers, dmin, dmax


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)

    tickers, dmin, dmax = _collect_watchlist()

    rows = []
    gaps_rows = []

    for tic in tickers:
        df = _load_close_any(tic)

        if df is None or df.empty:
            rows.append(
                {
                    "ticker": tic,
                    "rows": 0,
                    "first_date": None,
                    "last_date": None,
                    "dup_dates": None,
                    "non_monotonic": None,
                    "missing_bd_days": None,
                    "has_close_on_window": False,
                    "covers_targets_window": False,
                    "notes": "no_prices_file",
                }
            )
            continue

        # Basic hygiene
        df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
        rows_count = int(df.shape[0])
        first_date = df["date"].iloc[0] if rows_count else None
        last_date = df["date"].iloc[-1] if rows_count else None

        # Duplicates / monotonic
        dup_dates = int(df["date"].duplicated().sum())
        non_mono = bool((df["date"].diff().dt.total_seconds().fillna(0) < 0).any())

        # Gaps over entire file
        gaps = _business_day_gaps(df["date"])
        gaps_count = len(gaps)

        # Coverage vs target window
        window = df[(df["date"] >= dmin) & (df["date"] <= dmax)].copy()
        has_close_on_window = not window.empty
        covers_targets_window = bool((first_date <= dmin) and (last_date >= dmax)) if rows_count else False

        # Record gaps (limit to top 50 per ticker so CSV stays manageable)
        top_gaps = gaps[:50]
        for g in top_gaps:
            gaps_rows.append({"ticker": tic, "missing_bd_date": pd.Timestamp(g).date()})

        # Pass/fail PIT-ish check: must be monotonic, no dups, and have data in the window
        pass_pit = (dup_dates == 0) and (not non_mono) and has_close_on_window

        notes = []
        if dup_dates > 0:
            notes.append("dup_dates")
        if non_mono:
            notes.append("non_monotonic")
        if gaps_count > TRADING_DAYS_PER_WEEK * 4:
            notes.append("many_gaps")
        if not has_close_on_window:
            notes.append("no_data_in_targets_window")
        if not covers_targets_window:
            notes.append("partial_window")

        rows.append(
            {
                "ticker": tic,
                "rows": rows_count,
                "first_date": (None if first_date is None else str(pd.Timestamp(first_date).date())),
                "last_date": (None if last_date is None else str(pd.Timestamp(last_date).date())),
                "dup_dates": dup_dates,
                "non_monotonic": non_mono,
                "missing_bd_days": gaps_count,
                "has_close_on_window": has_close_on_window,
                "covers_targets_window": covers_targets_window,
                "pass_pit": pass_pit,
                "notes": ";".join(notes) if notes else "",
            }
        )

    df_sum = pd.DataFrame(rows)
    df_gaps = pd.DataFrame(gaps_rows)

    df_sum.to_csv(OUT_SUMMARY, index=False)
    df_gaps.to_csv(OUT_GAPS, index=False)

    passed = int(df_sum["pass_pit"].sum()) if not df_sum.empty and "pass_pit" in df_sum else 0
    total = int(df_sum.shape[0])
    out = {
        "out_summary_csv": str(OUT_SUMMARY),
        "out_gaps_csv": str(OUT_GAPS),
        "tickers": total,
        "pass_pit": passed,
        "fail_pit": total - passed,
        "targets_window": {
            "min": str(pd.Timestamp(dmin).date()),
            "max": str(pd.Timestamp(dmax).date()),
        },
        "git_sha8": _git_sha_short(),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

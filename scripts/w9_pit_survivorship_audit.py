# scripts/w9_pit_survivorship_audit.py
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(r"F:\Projects\trading_stack_py")
REPORTS = ROOT / "reports"
DATA = ROOT / "data" / "prices"
META = ROOT / "data" / "meta"

TARGETS_CSV = REPORTS / "wk11_blend_targets.csv"  # universe & dates anchor
DELIST_CSV = META / "delistings.csv"  # optional: ticker,delist_date
OUT_SCHEMA = REPORTS / "pit_schema_audit.csv"
OUT_MONO = REPORTS / "pit_monotonic_audit.csv"
OUT_SURV = REPORTS / "universe_survivorship_audit.csv"
OUT_DIAG = REPORTS / "w9_diag.json"

PIT_SUSPECT_TOKENS = [
    "adj",
    "adjusted",
    "future",
    "label",
    "target",
    "y_",
    "t+1",
    "t+2",
    "fwd",
    "lookahead",
    "leak",
    "prediction",
    "pred",
    "alpha",
    "signal",
]

DATE_CANDS = ["date", "dt", "timestamp", "asof", "as_of", "trading_day"]
CLOSE_CANDS = ["close", "px_close", "price", "last", "close_price"]
OPEN_CANDS = ["open", "px_open", "open_price"]
HIGH_CANDS = ["high", "px_high", "high_price"]
LOW_CANDS = ["low", "px_low", "low_price"]
VOL_CANDS = ["volume", "vol", "qty", "shares", "turnover"]


def _pick(cols, cands):
    low = {c.lower(): c for c in cols}
    for k in cands:
        if k in low:
            return low[k]
    for c in cols:
        lc = c.lower().replace(" ", "").replace("-", "_")
        for k in cands:
            if lc == k.replace(" ", "").replace("-", "_"):
                return c
    return None


def _business_gap_count(dts: pd.Series) -> int:
    if dts.empty:
        return 0
    dts = pd.to_datetime(dts).dropna().sort_values().unique()
    if len(dts) < 2:
        return 0
    # count gaps of > 5 calendar days as "likely missing window" (tolerate weekends/holidays)
    gaps = 0
    for i in range(1, len(dts)):
        delta = (dts[i] - dts[i - 1]).days
        if delta > 5:
            gaps += 1
    return int(gaps)


def _load_targets_universe():
    if not TARGETS_CSV.exists():
        raise SystemExit(f"Missing {TARGETS_CSV}. Run W11 first.")
    df = pd.read_csv(TARGETS_CSV)
    dcol = _pick(df.columns, DATE_CANDS)
    tcol = _pick(df.columns, ["ticker", "symbol", "name"])
    if not dcol or not tcol:
        raise SystemExit("wk11_blend_targets.csv missing date/ticker columns.")
    df[dcol] = pd.to_datetime(df[dcol], errors="coerce").dt.date
    dates = sorted(set(df[dcol].dropna()))
    last = max(dates) if dates else None
    universe_by_date = (
        df[[dcol, tcol]]
        .dropna()
        .astype({tcol: str})
        .groupby(dcol)[tcol]
        .agg(lambda s: sorted(set(s.tolist())))
        .reset_index()
        .rename(columns={dcol: "date", tcol: "tickers"})
    )
    return dates, last, universe_by_date


def _load_delist_map():
    if not DELIST_CSV.exists():
        return {}
    try:
        dd = pd.read_csv(DELIST_CSV)
        t = _pick(dd.columns, ["ticker", "symbol", "name"])
        d = _pick(dd.columns, DATE_CANDS)
        if not t or not d:
            return {}
        dd[d] = pd.to_datetime(dd[d], errors="coerce").dt.date
        return {str(r[t]): r[d] for _, r in dd.iterrows() if pd.notna(r[d])}
    except Exception:
        return {}


def _suspect_cols(cols):
    bad = []
    for c in cols:
        lc = c.lower()
        if any(tok in lc for tok in PIT_SUSPECT_TOKENS):
            bad.append(c)
    return bad


def _read_price_file(ticker: str):
    p = DATA / f"{ticker}.parquet"
    if not p.exists():
        return None, {"issue": "no_parquet"}
    try:
        x = pd.read_parquet(p)
        d = _pick(x.columns, DATE_CANDS)
        c = _pick(x.columns, CLOSE_CANDS)
        if not d or not c:
            return x, {"issue": "no_date_or_close", "columns": list(x.columns)}
        df = x[[d, c]].copy()
        df.columns = ["date", "close"]
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")
        return df, None
    except Exception as e:
        return None, {"issue": "read_error", "err": str(e)}


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    dates, last, univ_by_date = _load_targets_universe()
    delist_map = _load_delist_map()

    schema_rows = []
    mono_rows = []

    # Build quick lookup for survivorship audit
    want_last = (
        set(univ_by_date.loc[univ_by_date["date"] == last, "tickers"].iloc[0])
        if last
        else set()
    )

    have_last = set()
    for tic in sorted(want_last):
        df, err = _read_price_file(tic)
        # schema-level info
        p = DATA / f"{tic}.parquet"
        exists = p.exists()
        extra_cols = []
        has_date = has_close = False
        nrows = 0
        min_d = max_d = None

        if err:
            schema_rows.append(
                {
                    "ticker": tic,
                    "has_parquet": exists,
                    "has_date": False,
                    "has_close": False,
                    "rows": 0,
                    "min_date": None,
                    "max_date": None,
                    "pit_suspect_cols": ";".join([]),
                    "issue": err.get("issue", "unknown"),
                }
            )
            continue

        # when ok
        nrows = int(df.shape[0])
        has_date = True
        has_close = True
        min_d = df["date"].min().date() if nrows > 0 else None
        max_d = df["date"].max().date() if nrows > 0 else None

        # PIT suspect columns (check original file columns)
        try:
            raw = pd.read_parquet(p)
            extra_cols = _suspect_cols(raw.columns)
        except Exception:
            extra_cols = []

        schema_rows.append(
            {
                "ticker": tic,
                "has_parquet": exists,
                "has_date": has_date,
                "has_close": has_close,
                "rows": nrows,
                "min_date": min_d,
                "max_date": max_d,
                "pit_suspect_cols": ";".join(extra_cols),
                "issue": None,
            }
        )

        # monotonic / duplicates / gaps
        dts = df["date"]
        is_sorted = dts.is_monotonic_increasing
        dup_ct = int(dts.duplicated().sum())
        gap_ct = _business_gap_count(dts)

        mono_rows.append(
            {
                "ticker": tic,
                "rows": nrows,
                "is_monotonic_increasing": bool(is_sorted),
                "duplicate_dates": dup_ct,
                "gap_windows_gt5d": gap_ct,
                "min_date": min_d,
                "max_date": max_d,
            }
        )

        # survivorship presence at last day (exact match or prior day fallback)
        if last:
            m = df[df["date"] == pd.to_datetime(last)]
            if not m.empty:
                have_last.add(tic)
            else:
                # tolerate prior trading day coverage
                prev = df[df["date"] < pd.to_datetime(last)]
                if not prev.empty:
                    have_last.add(tic)

    # write schema & mono
    pd.DataFrame(schema_rows).to_csv(OUT_SCHEMA, index=False)
    pd.DataFrame(mono_rows).to_csv(OUT_MONO, index=False)

    # survivorship over full date range
    surv_rows = []
    for _, r in univ_by_date.iterrows():
        d = r["date"]
        ts = r["tickers"]
        want = set(ts)
        have = set()
        for tic in want:
            df, err = _read_price_file(tic)
            ok = False
            if not err and df is not None and not df.empty:
                if not df[df["date"] == pd.to_datetime(d)].empty:
                    ok = True
                else:
                    prev = df[df["date"] < pd.to_datetime(d)]
                    ok = not prev.empty
            have = have.union({tic}) if ok else have
        missing = sorted(list(want - have))
        # if delist map exists, tag a reason hint
        hints = []
        for tic in missing:
            dd = delist_map.get(tic)
            if dd and dd <= d:
                hints.append(f"{tic}:delisted@{dd}")
        surv_rows.append(
            {
                "date": d,
                "expected_universe": len(want),
                "coverage_ok": len(have),
                "missing": len(missing),
                "missing_sample": ";".join(missing[:5]),
                "delist_hints": ";".join(hints[:5]),
            }
        )
    pd.DataFrame(surv_rows).to_csv(OUT_SURV, index=False)

    out = {
        "as_of": str(last),
        "universe_lastday": len(want_last),
        "have_lastday_any_hist": len(have_last),
        "schema_csv": str(OUT_SCHEMA),
        "monotonic_csv": str(OUT_MONO),
        "survivorship_csv": str(OUT_SURV),
    }
    OUT_DIAG.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

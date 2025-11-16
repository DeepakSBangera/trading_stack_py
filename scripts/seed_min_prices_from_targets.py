# scripts/seed_min_prices_from_targets.py
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"F:\Projects\trading_stack_py")
REPORTS = ROOT / "reports"
DATA = ROOT / "data" / "prices"
TARGETS = REPORTS / "wk11_blend_targets.csv"
OUT_DIAG = REPORTS / "seed_min_prices_diag.json"

# knobs (synthetic generator)
BASE_PRICE = 100.0
ANNUAL_VOL = 0.18  # ~18% ann vol
DAILY_VOL = ANNUAL_VOL / (252.0**0.5)
DRIFT_DAILY = 0.00
VOLUME_BASE = 300_000
RANDOM_SEED = 12345  # deterministic

DATE_CANDS = ["date", "dt", "trading_day", "asof", "as_of"]
TICKER_CANDS = ["ticker", "symbol", "name", "secid", "instrument"]


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


def main():
    if not TARGETS.exists():
        raise SystemExit(f"Missing {TARGETS} (run W11 first).")

    df = pd.read_csv(TARGETS)
    dcol = _pick(df.columns, DATE_CANDS)
    tcol = _pick(df.columns, TICKER_CANDS)
    if not dcol or not tcol:
        raise SystemExit("wk11_blend_targets.csv missing date/ticker.")
    df[dcol] = pd.to_datetime(df[dcol], errors="coerce").dt.date
    dates = sorted(set(df[dcol].dropna()))
    tickers = sorted(set(df[tcol].astype(str)))

    if not dates or not tickers:
        raise SystemExit("No dates or tickers detected in W11 targets.")

    rng = np.random.default_rng(RANDOM_SEED)
    DATA.mkdir(parents=True, exist_ok=True)

    created, skipped = 0, 0
    for tic in tickers:
        p = DATA / f"{tic}.parquet"
        if p.exists():
            skipped += 1
            continue

        # synthetic geometric RW for CLOSE; then derive OHLC around it
        n = len(dates)
        rets = rng.normal(loc=DRIFT_DAILY, scale=DAILY_VOL, size=n)
        close = [BASE_PRICE]
        for r in rets[1:]:
            close.append(close[-1] * math.exp(r))
        close = np.array(close)

        # simple OHLC band around close (±0.6% intraday band)
        band = np.maximum(0.006 * close, 0.01)
        open_ = np.roll(close, 1)
        open_[0] = close[0]
        high = np.maximum.reduce([open_, close, close + band])
        low = np.minimum.reduce([open_, close, close - band])
        vol = (VOLUME_BASE * (1.0 + 0.1 * rng.standard_normal(n))).astype(int)

        out = pd.DataFrame(
            {
                "date": pd.to_datetime(dates),
                "open": open_.round(6),
                "high": high.round(6),
                "low": low.round(6),
                "close": close.round(6),
                "volume": vol,
            }
        )
        out.to_parquet(p, index=False)
        created += 1

    OUT_DIAG.write_text(
        json.dumps(
            {
                "tickers_total": len(tickers),
                "dates_total": len(dates),
                "created": created,
                "skipped_existing": skipped,
                "data_dir": str(DATA),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "created": created,
                "skipped_existing": skipped,
                "tickers": len(tickers),
                "dates": len(dates),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

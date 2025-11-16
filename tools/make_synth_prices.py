# tools/make_synth_prices.py
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def business_days(start, end):
    return pd.date_range(start=start, end=end, freq="B")


def make_names(n):
    return [f"S{str(i + 1).zfill(3)}.SYN" for i in range(n)]


def synth_prices(
    dates, mu_annual=0.10, vol_annual=0.25, n_sectors=5, symbols=None, seed=42
):
    rng = np.random.default_rng(seed)
    n = len(symbols)
    # Assign sectors
    sectors = rng.integers(0, n_sectors, size=n)
    # Market & sector factors
    dt = 1 / 252.0
    mu_dt = mu_annual * dt
    vol_m = 0.15 / np.sqrt(252)  # market daily vol
    vol_s = 0.10 / np.sqrt(252)  # sector daily vol
    # Factor paths
    T = len(dates)
    mkt = rng.normal(mu_dt, vol_m, size=T)
    sect = {k: rng.normal(0.0, vol_s, size=T) for k in range(n_sectors)}
    # Symbol params
    beta_m = rng.normal(1.0, 0.2, size=n)
    beta_s = rng.normal(1.0, 0.3, size=n)
    vol_id = rng.uniform(0.10, 0.30, size=n) / np.sqrt(252)
    mu_i = rng.normal(mu_annual, 0.03, size=n) * dt
    s0 = rng.uniform(80, 150, size=n)

    all_frames = {}
    for j, sym in enumerate(symbols):
        eps = rng.normal(0.0, vol_id[j], size=T)
        r = beta_m[j] * mkt + beta_s[j] * sect[sectors[j]] + mu_i[j] + eps
        # prices via GBM-ish compounding
        close = s0[j] * np.exp(np.cumsum(r))
        # fake OHLC + volume
        # open is yest close + tiny noise; high/low via intraday range
        oc_noise = rng.normal(0, 0.001, size=T)
        open_ = np.concatenate([[close[0]], close[:-1]]) * (1 + oc_noise)
        rng_range = rng.uniform(0.002, 0.02, size=T)
        high = np.maximum(open_, close) * (
            1 + rng_range * rng.uniform(0.3, 1.0, size=T)
        )
        low = np.minimum(open_, close) * (1 - rng_range * rng.uniform(0.3, 1.0, size=T))
        vol = (
            rng.lognormal(mean=12.0, sigma=0.3, size=T) * (1 + 2 * np.abs(r))
        ).astype(np.int64)

        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
            index=dates,
        )
        all_frames[sym] = df
    return all_frames, sectors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--symbols", type=int, default=150, help="number of synthetic symbols"
    )
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--end", default=pd.Timestamp.today().date().isoformat())
    ap.add_argument("--sectors", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--root", default="data")
    args = ap.parse_args()

    root = Path(args.root)
    dates = business_days(args.start, args.end)
    names = make_names(args.symbols)

    frames, sectors = synth_prices(
        dates, n_sectors=args.sectors, symbols=names, seed=args.seed
    )

    out_dir = root / "raw" / "synth" / "daily" / "SYN"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for sym, df in frames.items():
        p = out_dir / f"{sym.replace('.SYN', '')}.parquet"
        df.to_parquet(p)
        rows.append(
            {
                "symbol": sym,
                "exchange": "SYN",
                "sector": int(sectors[int(sym[1:4]) - 1]),
            }
        )

    # Write a universe CSV at repo root for convenience
    uni = pd.DataFrame(rows)
    uni_path = Path("my_universe.csv")
    uni.to_csv(uni_path, index=False)

    print(
        json.dumps(
            {
                "ok": True,
                "parquet_dir": str(out_dir),
                "symbols": len(rows),
                "universe_csv": str(uni_path),
                "date_first": str(dates[0].date()),
                "date_last": str(dates[-1].date()),
                "points": len(dates),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

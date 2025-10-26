import argparse
import pathlib

import numpy as np
import pandas as pd


def rng_series(dates, start=100.0, vol=0.02, seed=0):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0, vol / np.sqrt(252), size=len(dates))
    price = start * np.exp(np.cumsum(rets))
    return price


def make_prices(symbol, start, end, seed):
    dates = pd.date_range(start, end, freq="B")
    base = rng_series(dates, start=100.0, vol=0.22, seed=seed)
    hi = base * (1.0 + 0.005)
    lo = base * (1.0 - 0.005)
    vol = np.maximum(100000, (np.abs(np.diff(np.r_[0, base])) * 100000).astype(int))
    df = pd.DataFrame(
        {
            "date": dates,
            "symbol": symbol,
            "open": base,
            "high": hi,
            "low": lo,
            "close": base,
            "adj_close": base,
            "volume": vol,
        }
    )
    return df


def make_fundamentals(symbol, start, end, seed):
    rng = np.random.default_rng(seed)
    # Explicit quarterly *end* frequency (DEC year-end) to avoid deprecation warnings
    qdates = pd.date_range(start, end, freq="QE-DEC")
    sales = rng.uniform(100, 200, size=len(qdates)).cumsum()
    margin = rng.normal(0.18, 0.02, size=len(qdates)).clip(0.05, 0.6)
    netinc = sales * margin
    df = pd.DataFrame(
        {
            "asof": qdates,  # point-in-time date
            "symbol": symbol,
            "sales": sales,
            "net_income": netinc,
            "margin": margin,
        }
    )
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tickers",
        default="RELIANCE.NS,HDFCBANK.NS,INFY.NS,ICICIBANK.NS,TCS.NS,AXISBANK.NS,LT.NS,SBIN.NS,BHARTIARTL.NS,ITC.NS",
    )
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--outdir", default="data_synth")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = pathlib.Path(args.outdir)
    prices_dir = root / "prices"
    funda_dir = root / "fundamentals"
    prices_dir.mkdir(parents=True, exist_ok=True)
    funda_dir.mkdir(parents=True, exist_ok=True)

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    for i, sym in enumerate(tickers):
        p = make_prices(sym, args.start, args.end, seed=args.seed + i)
        f = make_fundamentals(sym, args.start, args.end, seed=args.seed + i)

        # enforce dtypes
        p["date"] = pd.to_datetime(p["date"])
        f["asof"] = pd.to_datetime(f["asof"])
        for c in ["open", "high", "low", "close", "adj_close"]:
            p[c] = pd.to_numeric(p[c], errors="coerce")
        p["volume"] = pd.to_numeric(p["volume"], errors="coerce").astype("Int64")

        # filenames
        pf = prices_dir / f"{sym.replace('.', '_')}.parquet"
        ff = funda_dir / f"{sym.replace('.', '_')}_fundamentals.parquet"

        p.to_parquet(pf, compression="snappy", index=False)
        f.to_parquet(ff, compression="snappy", index=False)

    print(f"✓ Wrote {len(tickers)} price files to {prices_dir} and fundamentals to {funda_dir}")


if __name__ == "__main__":
    main()

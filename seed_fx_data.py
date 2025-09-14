# Generate realistic(ish) FX OHLCV CSVs for symbols in data/universe/watchlist_fx.csv
# Columns: date,open,high,low,close,volume (volume=0 for FX)
import os, hashlib
import numpy as np
import pandas as pd

WATCHLIST = r"data/universe/watchlist_fx.csv"
OUTDIR = r"data/csv"
N_DAYS = 750  # ~3 years of business days

def make_series(seed, n=N_DAYS, spot=75.0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)

    # Daily returns: very low drift, low vol (typical major FX)
    rets = rng.normal(loc=0.0001, scale=0.005, size=n)  # ~1bp drift, ~50bp stdev
    close = spot * (1 + rets).cumprod()

    # Build OHLC around close with small intraday ranges
    # Range ~ 10–30 bps typical: scale with price
    intraday = np.abs(rng.normal(0.0, 0.0015, size=n)) * close  # ~15 bps
    open_ = np.r_[close[0], close[:-1]]  # previous close as open
    high = np.maximum(open_, close) + intraday
    low  = np.minimum(open_, close) - intraday

    # FX "volume" often not meaningful at daily frequency → set to zero
    vol = np.zeros(n, dtype=int)

    return pd.DataFrame({
        "date": dates,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": vol
    })

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    wl = pd.read_csv(WATCHLIST)["symbol"].dropna().astype(str).tolist()
    for sym in wl:
        # Stable, per-symbol seed so files are reproducible
        seed = int(hashlib.sha1(sym.encode()).hexdigest(), 16) % (2**32)
        # Rough base level per pair (keeps scales different)
        base = 60.0 + (seed % 50)  # e.g., USDINR around 60–110
        df = make_series(seed, spot=float(base))
        fp = os.path.join(OUTDIR, f"{sym}.csv")
        df.to_csv(fp, index=False)
        print("Wrote", fp)

if __name__ == "__main__":
    main()

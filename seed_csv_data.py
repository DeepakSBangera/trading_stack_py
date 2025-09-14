# Generate synthetic OHLCV CSVs for symbols in watchlist.csv (for offline testing)
import hashlib
import os

import numpy as np
import pandas as pd

WATCHLIST = r"data/universe/watchlist.csv"
OUTDIR = r"data/csv"
N_DAYS = 120  # business days


def make_series(seed, n=N_DAYS, start_price=250.0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    rets = rng.normal(loc=0.0008, scale=0.02, size=n)  # small drift, some vol
    close = start_price * (1 + rets).cumprod()
    # simple OHLCV from close
    atr_noise = np.abs(rng.normal(0.0, 0.01, size=n)) * close
    high = close + atr_noise
    low = close - atr_noise
    open_ = np.r_[close[0], close[:-1]]  # prev close as open
    vol = rng.integers(150_000, 450_000, size=n)
    df = pd.DataFrame(
        {"date": dates, "open": open_, "high": high, "low": low, "close": close, "volume": vol}
    )
    return df


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    wl = pd.read_csv(WATCHLIST)["symbol"].dropna().astype(str).tolist()
    for sym in wl:
        seed = int(hashlib.sha1(sym.encode()).hexdigest(), 16) % (2**32)
        base = 200 + (seed % 600)  # different starting levels
        df = make_series(seed, start_price=float(base))
        fp = os.path.join(OUTDIR, f"{sym}.csv")
        df.to_csv(fp, index=False)
        print("Wrote", fp)


if __name__ == "__main__":
    main()

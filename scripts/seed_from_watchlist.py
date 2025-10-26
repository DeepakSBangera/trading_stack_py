# Generate synthetic OHLCV for any watchlist (FX, commodities, equities) into data/csv/
# Usage examples:
#   python seed_from_watchlist.py --watchlist data/universe/watchlist_fx.csv
#   python seed_from_watchlist.py --watchlist data/universe/watchlist_cmdty.csv --days 1500
import argparse
import hashlib
import os

import numpy as np
import pandas as pd


def detect_asset_type(sym: str) -> str:
    if sym.endswith("=X"):
        return "fx"
    if sym.endswith("=F"):
        return "cmdty"
    return "equity"


def base_level(sym: str, asset: str) -> float:
    # Stable per-symbol base via hash; tweak per asset class
    h = int(hashlib.sha1(sym.encode()).hexdigest(), 16) % 10_000
    if asset == "fx":
        # keep bases in realistic zones; rough heuristics
        if "JPY" in sym:  # e.g., USDJPY ~ 110
            return 80 + (h % 80)  # 80–159
        if "INR" in sym:  # USDINR ~ 70-100
            return 60 + (h % 60)  # 60–119
        # majors ~ 0.7–1.6
        return 0.7 + (h % 90) / 100.0  # 0.7–1.59
    if asset == "cmdty":
        # very rough commodity levels
        return 20 + (h % 300)  # 20–319
    # equities: 100–700
    return 100 + (h % 600)


def make_series(seed: int, n: int, spot: float, asset: str) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)

    if asset == "fx":
        drift, vol = 0.0001, 0.005  # ~1bp drift, 50bp daily vol
        intraday_scale = 0.0015  # ~15bp intraday range
        volume = np.zeros(n, dtype=int)
    elif asset == "cmdty":
        drift, vol = 0.0002, 0.01  # 2bp drift, 1% vol
        intraday_scale = 0.01
        volume = rng.integers(5_000, 50_000, size=n)
    else:  # equity
        drift, vol = 0.0004, 0.02  # 4bp drift, 2% vol
        intraday_scale = 0.01
        volume = rng.integers(150_000, 450_000, size=n)

    rets = rng.normal(loc=drift, scale=vol, size=n)
    close = spot * (1 + rets).cumprod()

    # OHLC from close
    open_ = np.r_[close[0], close[:-1]]
    intraday = np.abs(rng.normal(0.0, intraday_scale, size=n)) * close
    high = np.maximum(open_, close) + intraday
    low = np.minimum(open_, close) - intraday

    df = pd.DataFrame(
        {"date": dates, "open": open_, "high": high, "low": low, "close": close, "volume": volume}
    )
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--watchlist",
        default=r"data/universe/watchlist.csv",
        help="Path to watchlist CSV with a 'symbol' column",
    )
    ap.add_argument("--outdir", default=r"data/csv", help="Output directory for generated CSVs")
    ap.add_argument("--days", type=int, default=750, help="Number of business days to generate")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    wl = pd.read_csv(args.watchlist)["symbol"].dropna().astype(str).tolist()
    if not wl:
        print(f"[ERR] No symbols found in {args.watchlist}")
        return

    for sym in wl:
        asset = detect_asset_type(sym)
        seed = int(hashlib.sha1(sym.encode()).hexdigest(), 16) % (2**32)
        spot = float(base_level(sym, asset))
        df = make_series(seed, n=args.days, spot=spot, asset=asset)
        fp = os.path.join(args.outdir, f"{sym}.csv")
        df.to_csv(fp, index=False)
        print(f"Wrote {fp}  [{asset}, base≈{spot:.2f}]")


if __name__ == "__main__":
    main()

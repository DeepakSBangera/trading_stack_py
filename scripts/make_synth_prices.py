import argparse

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--periods", type=int, default=800)  # ~3.2y of trading days
    ap.add_argument("--start_price", type=float, default=2500.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dates = pd.date_range(args.start, periods=args.periods, freq=BDay())
    # fat-tailed daily shocks; small positive drift
    shocks = 0.0002 + 0.01 * np.random.standard_t(df=6, size=args.periods)
    prices = args.start_price * np.exp(np.cumsum(shocks))
    df = pd.DataFrame({"date": dates.date, "close": prices})
    df.to_csv(args.out, index=False)
    print(f"Saved synthetic series: {len(df)} rows -> {args.out}")


if __name__ == "__main__":
    main()

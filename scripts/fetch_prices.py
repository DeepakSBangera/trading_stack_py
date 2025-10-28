import argparse
import time

import pandas as pd
import yfinance as yf


def try_download(ticker, start, end, attempts=6, sleep_s=10):
    last_err = None
    for _k in range(1, attempts + 1):
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                progress=False,
                threads=False,
                interval="1d",
            )
            if not df.empty:
                return df
        except Exception as e:
            last_err = e
        time.sleep(sleep_s)  # backoff
    if last_err:
        raise last_err
    return pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--attempts", type=int, default=6)
    ap.add_argument("--sleep", type=int, default=10)
    args = ap.parse_args()

    df = try_download(
        args.ticker, args.start, args.end, attempts=args.attempts, sleep_s=args.sleep
    )
    if df.empty:
        raise SystemExit(
            f"No data for {args.ticker}. Possibly rate-limited; try later."
        )
    df = df[["Close"]].reset_index()
    df.columns = ["date", "close"]
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()

# Data adapters (yfinance or CSV) + lightweight I/O helpers
from pathlib import Path

import pandas as pd
import yfinance as yf


def fetch_ohlcv(symbol, start):
    """Single-symbol fetch (kept for compatibility)."""
    df = yf.download(symbol, start=start, progress=False, auto_adjust=False, threads=False)
    if df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"]).set_index(
            pd.DatetimeIndex([])
        )
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    df.index.name = "date"
    return df


def fetch_ohlcv_batch(symbols, start):
    """
    Batch fetch using one Yahoo call -> fewer requests -> fewer rate-limit errors.
    Returns dict: {symbol: DataFrame or empty DataFrame}
    """
    if not symbols:
        return {}
    tickers_str = " ".join(symbols)
    df = yf.download(
        tickers=tickers_str,
        start=start,
        progress=False,
        auto_adjust=False,
        threads=False,  # safer with rate limits
        group_by="ticker",
        interval="1d",
    )

    out = {}
    if df is None or len(df) == 0:
        # all failed
        for s in symbols:
            out[s] = pd.DataFrame()
        return out

    # If multiple tickers, we get a MultiIndex columns: (TICKER, FIELD)
    if isinstance(df.columns, pd.MultiIndex):
        for s in symbols:
            if s in df.columns.get_level_values(0):
                sub = df[s].rename(columns=str.lower)
                cols = [c for c in ["open", "high", "low", "close", "volume"] if c in sub.columns]
                if cols:
                    sub = sub[cols]
                    sub.index.name = "date"
                    out[s] = sub.dropna(how="all")
                else:
                    out[s] = pd.DataFrame()
            else:
                out[s] = pd.DataFrame()
    else:
        # Single ticker case
        sub = df.rename(columns=str.lower)
        cols = [c for c in ["open", "high", "low", "close", "volume"] if c in sub.columns]
        if cols:
            sub = sub[cols]
            sub.index.name = "date"
            out[symbols[0]] = sub.dropna(how="all")
        else:
            out[symbols[0]] = pd.DataFrame()

    return out


def fetch_ohlcv_from_csv(symbol, csv_dir="data/csv"):
    """Load OHLCV from your own CSV files."""
    fp = Path(csv_dir) / f"{symbol}.csv"
    if not fp.exists():
        return pd.DataFrame()
    df = pd.read_csv(fp)
    df.columns = [c.lower() for c in df.columns]
    if "date" not in df.columns:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    cols = ["open", "high", "low", "close", "volume"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        return pd.DataFrame()
    return df[cols]


def save_parquet(df, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def load_watchlist(csv_path):
    return pd.read_csv(csv_path)["symbol"].dropna().astype(str).tolist()

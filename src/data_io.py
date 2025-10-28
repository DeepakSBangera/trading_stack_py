# Data adapters (yfinance or CSV) + lightweight I/O helpers
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf


def fetch_ohlcv(symbol: str, start: str) -> pd.DataFrame:
    df = yf.download(
        symbol, start=start, progress=False, auto_adjust=False, threads=False
    )
    if df.empty:
        return pd.DataFrame(
            columns=["date", "open", "high", "low", "close", "volume"]
        ).set_index(pd.DatetimeIndex([]))
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    df.index.name = "date"
    return df


def fetch_ohlcv_from_csv(symbol: str, csv_dir: str = "data/csv") -> pd.DataFrame:
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


def save_parquet(df: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def load_watchlist(csv_path: str) -> list[str]:
    return pd.read_csv(csv_path)["symbol"].dropna().astype(str).tolist()

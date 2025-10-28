# src/trading_stack_py/data_loader.py
from __future__ import annotations

import glob
import math
import os
import random
import time
from typing import Literal

import numpy as np
import pandas as pd

# yfinance is optional; only used if source='yahoo' or auto falls back to yahoo
try:
    import yfinance as yf  # type: ignore

    _HAS_YF = True
except Exception:
    _HAS_YF = False

CACHE_DIR = os.path.join("data", "csv")
os.makedirs(CACHE_DIR, exist_ok=True)

SourceT = Literal["auto", "local", "yahoo", "synthetic"]

# Simple alias map for common NSE tickers (fallbacks if primary fails)
ALIASES: dict[str, list[str]] = {
    "NIFTYBEES.NS": ["^NSEI"],  # ETF -> index fallback
}


# -------------------- cache helpers --------------------
def _cache_path(ticker: str) -> str:
    safe = ticker.replace("/", "_").replace("\\", "_").replace(".", "_")
    return os.path.join(CACHE_DIR, f"{safe}.csv")


def _read_cache(cp: str) -> pd.DataFrame | None:
    if not os.path.exists(cp):
        return None
    try:
        df = pd.read_csv(cp, parse_dates=["Date"])
        return df
    except Exception:
        df = pd.read_csv(cp)
        if "Date" not in df.columns:
            if "Datetime" in df.columns:
                df.rename(columns={"Datetime": "Date"}, inplace=True)
            elif df.columns and df.columns[0].lower() in {"date", "datetime", "index"}:
                df.rename(columns={df.columns[0]: "Date"}, inplace=True)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.dropna(subset=["Date"])
            return df if not df.empty else None
        return None


def _write_cache(cp: str, df: pd.DataFrame) -> None:
    if "Date" not in df.columns:
        df = df.reset_index().rename(columns={"index": "Date"})
    # Normalize Yahoo column casing
    ren = {c: c.strip().title() for c in df.columns}
    df = df.rename(columns=ren)
    df.to_csv(cp, index=False)


# -------------------- local CSV --------------------
def _find_local_csv(ticker: str) -> str | None:
    exact = _cache_path(ticker)
    if os.path.exists(exact):
        return exact
    base = ticker.replace(".", "_").lower()
    for c in glob.glob(os.path.join(CACHE_DIR, "*.csv")):
        if base in os.path.basename(c).lower():
            return c
    return None


def _load_local(ticker: str) -> pd.DataFrame | None:
    cp = _find_local_csv(ticker)
    if not cp:
        return None
    return _read_cache(cp)


# -------------------- synthetic --------------------
def _gen_synth(
    ticker: str, start: str = "2015-01-01", end: str | None = None
) -> pd.DataFrame:
    # If end is None, use fixed periods; else build between start and end.
    if end is None:
        idx = pd.date_range(start=start, periods=1500, freq="B")
    else:
        idx = pd.date_range(start=start, end=end, freq="B")

    n = len(idx)
    if n == 0:
        idx = pd.date_range(start="2019-01-01", periods=600, freq="B")
        n = len(idx)

    rng = np.random.default_rng(seed=abs(hash((ticker, start, end))) % (2**32))
    mu_annual = 0.15
    vol_annual = 0.22
    mu_daily = mu_annual / 252.0
    vol_daily = vol_annual / math.sqrt(252.0)

    rets = rng.normal(mu_daily, vol_daily, size=n)
    price0 = 100.0
    prices = price0 * np.exp(np.cumsum(rets))

    noise = rng.normal(0.0, 0.0015, size=n)
    open_ = prices * (1.0 + noise)
    # preliminary hi/lo around open/close; corrected below
    hi0 = np.maximum(open_, prices) * (1.0 + np.abs(rng.normal(0.002, 0.0015, n)))
    lo0 = np.minimum(open_, prices) * (1.0 - np.abs(rng.normal(0.002, 0.0015, n)))
    vol = rng.integers(1_000_000, 5_000_000, size=n)

    df = pd.DataFrame(
        {
            "Date": idx,
            "Open": open_,
            "High": hi0,
            "Low": lo0,
            "Close": prices,
            "Adj Close": prices,
            "Volume": vol,
        }
    )
    # Ensure OHLC invariants
    df["High"] = df[["Open", "Close", "High"]].max(axis=1)
    df["Low"] = df[["Open", "Close", "Low"]].min(axis=1)
    return df


# -------------------- yahoo --------------------
def _yahoo_download(ticker: str, start: str | None, end: str | None) -> pd.DataFrame:
    if not _HAS_YF:
        raise RuntimeError("yfinance not installed")
    df = yf.download(
        ticker, start=start, end=end, interval="1d", auto_adjust=False, progress=False
    )
    if df is None or df.empty:
        df = yf.Ticker(ticker).history(
            start=start, end=end, interval="1d", auto_adjust=False
        )
    if df is None or df.empty:
        raise RuntimeError(f"No data for {ticker}")
    if "Date" not in df.columns:
        df = df.reset_index()
    return df


def _fetch_with_backoff(
    ticker: str, start: str | None, end: str | None, retries: int = 6
) -> pd.DataFrame:
    base = 2.0
    for attempt in range(retries):
        try:
            return _yahoo_download(ticker, start, end)
        except Exception as e:
            msg = str(e)
            transient = any(
                k in msg
                for k in [
                    "Too Many Requests",
                    "rate",
                    "429",
                    "timed out",
                    "Connection aborted",
                ]
            )
            if transient:
                sleep_s = base**attempt + random.uniform(0.0, 1.0)
                time.sleep(min(60.0, sleep_s))
                continue
            raise
    return _yahoo_download(ticker, start, end)


# -------------------- public API --------------------
def get_prices(
    ticker: str,
    start: str | None = "2015-01-01",
    end: str | None = None,
    force_refresh: bool = False,
    source: SourceT = "auto",
) -> pd.DataFrame:
    """
    Return OHLCV daily dataframe with columns:
      Date, Open, High, Low, Close, Adj Close, Volume

    source:
      - 'auto' (default): try local CSV -> yahoo -> synthetic
      - 'local': only local CSV under data/csv
      - 'yahoo': only Yahoo Finance (with backoff/aliases)
      - 'synthetic': generated random-walk series
    """
    # 1) LOCAL
    if source in ("auto", "local"):
        if not force_refresh or source == "local":
            local = _load_local(ticker)
            if local is not None:
                return local

    # 2) YAHOO
    if source in ("auto", "yahoo") and _HAS_YF:
        try:
            df = _fetch_with_backoff(ticker, start, end)
            _write_cache(_cache_path(ticker), df)
            return df
        except Exception:
            for alt in ALIASES.get(ticker, []):
                try:
                    df_alt = _fetch_with_backoff(alt, start, end)
                    _write_cache(_cache_path(ticker), df_alt)
                    return df_alt
                except Exception:
                    pass
            if source == "yahoo":
                raise

    # 3) SYNTHETIC (always last resort)
    synth = _gen_synth(ticker, start=start or "2015-01-01", end=end)
    _write_cache(_cache_path(ticker), synth)
    return synth


def get_panel(tickers: list[str], **kwargs) -> dict[str, pd.DataFrame]:
    return {t: get_prices(t, **kwargs) for t in tickers}

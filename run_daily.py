# Daily trading driver: fetch -> features -> signals -> today's BUY list CSV
import os
import sys
from datetime import datetime

import pandas as pd
import yaml
from src import data_io
from src.signals import make_signals

cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config/config.yaml"
with open(cfg_path, encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

os.makedirs("reports", exist_ok=True)
os.makedirs("data/prices", exist_ok=True)
os.makedirs("data/features", exist_ok=True)

symbols = data_io.load_watchlist(CFG["data"]["universe_csv"])
start = CFG["data"].get("start", "2018-01-01")
source = CFG["data"].get("source", "yfinance")  # "yfinance" or "csv"
csv_dir = CFG["data"].get("csv_dir", "data/csv")  # used only if source == "csv"

rule_name = CFG["signals"].get("rule", "R1_trend_breakout_obv")
params = CFG["signals"].get("params", {})


def get_df_for_symbol(sym, yfin_cache=None):
    """Fetch df for one symbol based on source setting; try cache on failure."""
    if source == "yfinance":
        if yfin_cache is not None:
            df = yfin_cache.get(sym, pd.DataFrame())
        else:
            df = data_io.fetch_ohlcv(sym, start=start)
    elif source == "csv":
        df = data_io.fetch_ohlcv_from_csv(sym, csv_dir=csv_dir)
    else:
        raise ValueError(f"Unknown data source: {source}")

    if df is None or df.empty:
        # Fallback to cached parquet if it exists
        cache_fp = f"data/prices/{sym}.parquet"
        if os.path.exists(cache_fp):
            try:
                df = pd.read_parquet(cache_fp)
                print(f"[INFO] Using cached data for {sym}")
            except Exception:
                pass
    return df if df is not None else pd.DataFrame()


all_buys = []

if source == "yfinance":
    # Batch fetch to reduce rate-limit issues
    batch = data_io.fetch_ohlcv_batch(symbols, start=start)
    for sym in symbols:
        df = get_df_for_symbol(sym, yfin_cache=batch)
        if df.empty:
            print(f"[WARN] No data for {sym}")
            continue

        df.to_parquet(f"data/prices/{sym}.parquet")
        sigdf = make_signals(df, params, rule_name)
        sigdf.to_parquet(f"data/features/{sym}.parquet")

        last = sigdf.iloc[-1].copy()
        if int(last.get("buy", 0)) == 1:
            all_buys.append(
                {
                    "symbol": sym,
                    "close": float(last["close"]),
                    "atr": float(last["atr"]),
                    "score": int(last["score"]),
                }
            )
else:
    # CSV source: iterate one by one
    for sym in symbols:
        df = get_df_for_symbol(sym)
        if df.empty:
            print(f"[WARN] No data for {sym}")
            continue

        df.to_parquet(f"data/prices/{sym}.parquet")
        sigdf = make_signals(df, params, rule_name)
        sigdf.to_parquet(f"data/features/{sym}.parquet")

        last = sigdf.iloc[-1].copy()
        if int(last.get("buy", 0)) == 1:
            all_buys.append(
                {
                    "symbol": sym,
                    "close": float(last["close"]),
                    "atr": float(last["atr"]),
                    "score": int(last["score"]),
                }
            )

today = datetime.today().date().isoformat()
buylist_path = f"reports/buylist_{today}.csv"

# Graceful handling: if empty, still write a CSV with headers
cols = ["symbol", "close", "atr", "score"]
out_df = pd.DataFrame(all_buys, columns=cols)
if out_df.empty:
    out_df.to_csv(buylist_path, index=False)
else:
    out_df.sort_values(by=["score", "symbol"], ascending=[False, True]).to_csv(
        buylist_path, index=False
    )

print(f"Wrote {buylist_path}")

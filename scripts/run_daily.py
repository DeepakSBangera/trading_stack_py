# run_daily.py — safe when no data or no signals
from __future__ import annotations

import os
import time
from datetime import datetime

import pandas as pd
import yaml
from src.signals import make_signals

from src import data_io

with open("config/config.yaml", encoding="utf-8") as f:
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

all_buys: list[dict] = []
for sym in symbols:
    # polite pause (helps APIs; harmless for CSV)
    time.sleep(0.2)

    # Choose data source
    if source == "yfinance":
        df = data_io.fetch_ohlcv(sym, start=start)
    elif source == "csv":
        df = data_io.fetch_ohlcv_from_csv(sym, csv_dir=csv_dir)
    else:
        raise ValueError(f"Unknown data source: {source}")

    if df.empty:
        print(f"[WARN] No data for {sym}")
        continue

    # Cache raw & features (parquet)
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

# Write output safely even if empty
out_cols = ["symbol", "close", "atr", "score"]
out_df = pd.DataFrame(all_buys, columns=out_cols)

if out_df.empty:
    print("[INFO] No signals today (or no data). Writing empty buylist with headers.")
    out_df.to_csv(buylist_path, index=False)
else:
    out_df.sort_values(by=["score", "symbol"], ascending=[False, True]).to_csv(buylist_path, index=False)
print(f"Wrote {buylist_path}")

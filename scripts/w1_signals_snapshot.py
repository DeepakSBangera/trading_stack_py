import os
from datetime import datetime

import pandas as pd
import yaml

from src import data_io
from src.signals import make_signals


def main() -> None:
    with open("config/config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    symbols = data_io.load_watchlist(cfg["data"]["universe_csv"])
    source = cfg["data"].get("source", "yfinance")
    start = cfg["data"].get("start", "2018-01-01")
    rule_name = cfg["signals"].get("rule", "R1_trend_breakout_obv")
    params = cfg["signals"].get("params", {}) or {}

    rows = []
    for sym in symbols:
        if source == "csv":
            csv_dir = cfg["data"].get("csv_dir", "data/csv")
            df = data_io.fetch_ohlcv_from_csv(sym, csv_dir=csv_dir)
        else:
            df = data_io.fetch_ohlcv(sym, start=start)

        if df.empty or len(df) < 60:
            rows.append({"symbol": sym, "status": "no_data"})
            continue

        sig = make_signals(df, params, rule_name)
        last = sig.iloc[-1].copy()
        rows.append(
            {
                "symbol": sym,
                "buy": int(last.get("buy", 0)),
                "score": int(last.get("score", 0)),
                "close": float(last.get("close", float("nan"))),
                "atr": float(last.get("atr", float("nan"))),
                "rsi": float(last.get("rsi", float("nan"))),
                "sma_cross": int(last.get("sma_f", 0) > last.get("sma_s", 0)),
            }
        )

    out = pd.DataFrame(rows).sort_values(["buy", "score", "symbol"], ascending=[False, False, True])
    os.makedirs("reports", exist_ok=True)
    path = "reports/wk1_entry_exit_baseline.csv"
    out.to_csv(path, index=False)

    log = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "rule": rule_name,
        "params": params,
        "source": source,
        "n_symbols": len(symbols),
        "n_buys": int(out["buy"].sum()) if "buy" in out else 0,
    }
    with open("reports/w1_run_log.txt", "w", encoding="utf-8") as f:
        f.write(yaml.safe_dump(log, sort_keys=False))

    print(f"Wrote {path} with {log['n_buys']} buys")


if __name__ == "__main__":
    main()

# tools/build_factors.py
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_price_series(root: Path, source="synth", exchange="SYN"):
    base = root / "raw" / source / "daily" / exchange
    px = {}
    for fp in base.glob("*.parquet"):
        sym = fp.stem  # e.g., S001
        df = pd.read_parquet(fp)
        if "close" not in df.columns:
            continue
        s = pd.Series(df["close"].values, index=pd.to_datetime(df.index), name=sym)
        px[sym] = s.asfreq("B").ffill()
    if not px:
        return pd.DataFrame()
    return pd.DataFrame(px).sort_index()


def calc_factors(prices: pd.DataFrame):
    rets = prices.pct_change()
    panel = {}
    # 6m momentum (total return)
    panel["mom126"] = (
        (1 + rets).rolling(126).apply(lambda x: np.prod(1 + x) - 1, raw=False)
    )
    # 20d volatility
    panel["vol20"] = rets.rolling(20).std()
    # Simple quality proxy
    u = rets.clip(lower=0).rolling(30).std()
    d = (-rets).clip(lower=0).rolling(30).std()
    panel["quality"] = u - d
    return panel


def save_factor_marts(panel, root: Path):
    for name, df in panel.items():
        outdir = root / "marts" / "factors" / f"factor={name}"
        outdir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(outdir / "full.parquet")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--source", default="synth")  # synth | kite (later)
    ap.add_argument("--exchange", default="SYN")
    args = ap.parse_args()

    root = Path(args.root)
    prices = load_price_series(root, source=args.source, exchange=args.exchange)
    if prices.empty:
        print(
            json.dumps(
                {
                    "ok": False,
                    "reason": f"no parquet under {root}/raw/{args.source}/daily/{args.exchange}",
                },
                indent=2,
            )
        )
        return
    panel = calc_factors(prices)
    save_factor_marts(panel, root)
    print(
        json.dumps(
            {
                "ok": True,
                "factors": list(panel.keys()),
                "cols": prices.shape[1],
                "rows": prices.shape[0],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

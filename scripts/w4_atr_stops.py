from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
BT_BASE = ROOT / "reports" / "backtests"
CONFIG = ROOT / "config" / "w4_atr.yaml"
OUT = ROOT / "reports" / "wk4_atr_stops.csv"


def latest_backtest_dir(base: Path) -> Path:
    runs = sorted(
        [d for d in base.iterdir() if d.is_dir()], key=lambda p: p.stat().st_mtime
    )
    if not runs:
        raise SystemExit(
            "No backtest runs found. Run: python -m scripts.w2_backtest first."
        )
    return runs[-1]


def load_latest_run() -> tuple[pd.DataFrame, pd.Series]:
    d = latest_backtest_dir(BT_BASE)
    px = pd.read_csv(d / "px.csv", index_col=0, parse_dates=True)
    last_w = pd.read_csv(d / "last_weights.csv", index_col=0)["weight"]
    # Align to shared tickers
    last_w = last_w[last_w.index.intersection(px.columns)]
    last_w = last_w[last_w > 0].sort_values(ascending=False)
    return px, last_w


def atr_proxy(series: pd.Series, window: int = 14) -> pd.Series:
    """
    ATR proxy using absolute daily price change (works with Close-only data):
    TR_t = |P_t - P_{t-1}|
    ATR_t = EMA(TR, span=window)
    """
    tr = series.diff().abs()
    return tr.ewm(span=window, adjust=False).mean()


def main() -> None:
    if not CONFIG.exists():
        raise SystemExit(f"Config not found: {CONFIG}")

    with open(CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    w = int(cfg.get("atr_window", 14))
    k = float(cfg.get("atr_mult", 3.0))
    trailing = bool(cfg.get("use_trailing", True))

    px, last_w = load_latest_run()
    if last_w.empty:
        raise SystemExit("No non-zero weights in the latest run's last_weights.csv.")

    # Compute ATR proxy per asset (on price)
    atr_df = pd.DataFrame({c: atr_proxy(px[c], window=w) for c in last_w.index}).dropna(
        how="all"
    )

    # Build stop table for currently held names
    rows = []
    for c, wt in last_w.items():
        s = px[c].dropna()
        if s.empty:
            continue
        atr = atr_df[c].reindex(s.index).fillna(method="ffill")

        price = float(s.iloc[-1])
        atr_now = (
            float(atr.iloc[-1]) if not math.isnan(float(atr.iloc[-1])) else float("nan")
        )

        if trailing:
            # trailing stop from rolling max close
            roll_max = s.cummax()
            # ‘distance’ from recent high; suggested stop is rolling max - k*ATR
            stop_level = (
                float(roll_max.iloc[-1] - k * atr_now)
                if atr_now == atr_now
                else float("nan")
            )
        else:
            # fixed stop from current price
            stop_level = (
                float(price - k * atr_now) if atr_now == atr_now else float("nan")
            )

        rows.append(
            {
                "ticker": c,
                "weight": float(wt),
                "price": price,
                "atr_window": w,
                "atr_proxy": atr_now,
                "atr_mult": k,
                "use_trailing": trailing,
                "stop_level": stop_level,
                "stop_pct": (
                    (stop_level / price - 1.0)
                    if (price and stop_level == stop_level)
                    else float("nan")
                ),
            }
        )

    df = pd.DataFrame(rows).sort_values("weight", ascending=False)

    # Portfolio-level ATR estimate: weight the asset ATRs by weights (rough proxy)
    # Convert ATR in price units to percent-of-price first, then weight-average.
    pct_atr = []
    for _, r in df.iterrows():
        if r["atr_proxy"] == r["atr_proxy"] and r["price"]:
            pct_atr.append(r["weight"] * (r["atr_proxy"] / r["price"]))
    port_atr_pct = float(sum(pct_atr)) if pct_atr else float("nan")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    print("=== ATR Stops (Week 4) ===")
    print(f"Held names: {len(df)}")
    print(f"ATR window: {w}  |  ATR multiple: {k}  | trailing: {trailing}")
    print(
        f"Portfolio ATR (approx): {port_atr_pct * 100:.2f}% per day"
        if port_atr_pct == port_atr_pct
        else "Portfolio ATR: N/A"
    )
    print(f"Wrote: {OUT}")


if __name__ == "__main__":
    main()

# scripts/w4_voltarget_stops.py
from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"F:\Projects\trading_stack_py")
REPORTS = ROOT / "reports"
DATA = ROOT / "data" / "prices"

TARGETS_CSV = REPORTS / "wk11_blend_targets.csv"  # tickers + dates
VOLSTOPS_CSV = REPORTS / "wk4_voltarget_stops.csv"  # main output
THROTTLE_CSV = REPORTS / "dd_throttle_map.csv"  # DD → risk multiplier
KILL_SWITCH_YML = REPORTS / "kill_switch.yaml"  # policy knobs
DIAG_JSON = REPORTS / "w4_vol_diag.json"

# --- knobs (tune as needed) ---
LOOKBACK_DAYS_RET = 20  # realized vol window (trailing)
ATR_LEN = 14  # ATR window
STOP_ATR_MULT = 3.0  # classic ATR multiple
TARGET_VOL_ANN = 0.12  # 12% annualized; guidance
BASE_KELLY = 0.25  # base Kelly fraction


def _pick(cols, cands):
    low = {c.lower(): c for c in cols}
    for k in cands:
        if k in low:
            return low[k]
    for c in cols:
        lc = c.lower().replace(" ", "").replace("-", "_")
        for k in cands:
            if lc == k.replace(" ", "").replace("-", "_"):
                return c
    return None


def _load_targets_lastday() -> tuple[dt.date, list[str]]:
    df = pd.read_csv(TARGETS_CSV)
    dcol = _pick(df.columns, ["date", "dt", "trading_day", "asof", "as_of"])
    tcol = _pick(df.columns, ["ticker", "symbol", "name", "secid", "instrument"])
    if not dcol or not tcol:
        raise SystemExit("wk11_blend_targets.csv missing date/ticker columns.")
    df[dcol] = pd.to_datetime(df[dcol], errors="coerce").dt.date
    last = df[dcol].dropna().max()
    subs = df.loc[df[dcol] == last, tcol].astype(str).unique().tolist()
    if not subs:
        raise SystemExit("No tickers on last day in wk11_blend_targets.csv.")
    return last, subs


def _load_price_df(ticker: str) -> pd.DataFrame | None:
    p = DATA / f"{ticker}.parquet"
    if not p.exists():
        return None
    try:
        x = pd.read_parquet(p)
        cols = {c.lower(): c for c in x.columns}
        dcol = cols.get("date") or cols.get("dt")
        ocol = cols.get("open")
        hcol = cols.get("high")
        lcol = cols.get("low")
        ccol = cols.get("close") or cols.get("px_close") or cols.get("price")
        if dcol is None or ccol is None:
            return None
        x = x[[dcol] + [c for c in [ocol, hcol, lcol, ccol] if c is not None]].copy()
        x.columns = ["date"] + [c for c in ["open", "high", "low", "close"] if c in x.columns]
        x["date"] = pd.to_datetime(x["date"], errors="coerce")
        x = x.dropna(subset=["date"]).sort_values("date")
        return x
    except Exception:
        return None


def _compute_atr(df: pd.DataFrame, atr_len: int) -> pd.Series:
    # If OHLC available: classic ATR; else fallback to TrueRange ~ |close-close.shift(1)|
    if set(["high", "low", "close"]).issubset(df.columns):
        prev_close = df["close"].shift(1)
        tr = pd.concat(
            [
                (df["high"] - df["low"]).abs(),
                (df["high"] - prev_close).abs(),
                (df["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(atr_len, min_periods=max(2, atr_len // 2)).mean()
    else:
        tr = (df["close"] - df["close"].shift(1)).abs()
        atr = tr.rolling(atr_len, min_periods=max(2, atr_len // 2)).mean()
    return atr


def _ann_vol_from_returns(df: pd.DataFrame, lookback: int) -> float | None:
    if "close" not in df.columns:
        return None
    rets = df["close"].pct_change()
    tail = rets.dropna().tail(lookback)
    if tail.empty:
        return None
    vol = float(tail.std(ddof=0)) * math.sqrt(252.0)
    return vol if math.isfinite(vol) else None


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    diag = {
        "as_of": None,
        "tickers": 0,
        "with_data": 0,
        "rows": 0,
        "overall_vol_ann": None,
        "target_vol_ann": TARGET_VOL_ANN,
        "kelly_base": BASE_KELLY,
        "notes": [],
    }

    last_day, tickers = _load_targets_lastday()
    diag["as_of"] = str(last_day)
    diag["tickers"] = len(tickers)

    out_rows = []
    vols_ = []
    for tic in tickers:
        df = _load_price_df(tic)
        if df is None:
            diag["notes"].append(f"{tic}: no parquet/close")
            continue

        # keep data up to last_day
        df = df[df["date"] <= pd.to_datetime(last_day)]
        if df.empty or df.shape[0] < 5:
            diag["notes"].append(f"{tic}: insufficient history")
            continue

        df = df.tail(max(LOOKBACK_DAYS_RET * 3, ATR_LEN + 10)).copy()
        df["atr"] = _compute_atr(df, ATR_LEN)
        vol_ann = _ann_vol_from_returns(df, LOOKBACK_DAYS_RET)
        if vol_ann is not None:
            vols_.append(vol_ann)

        # last available row (<= last_day)
        r = df.iloc[-1]
        close = float(r["close"])
        atr = float(r["atr"]) if math.isfinite(float(r["atr"])) else float("nan")

        # stops (long/short lenses)
        long_stop = close - STOP_ATR_MULT * atr if math.isfinite(atr) else float("nan")
        short_stop = close + STOP_ATR_MULT * atr if math.isfinite(atr) else float("nan")

        out_rows.append(
            {
                "date": last_day,
                "ticker": tic,
                "close": round(close, 6),
                f"atr_{ATR_LEN}": round(atr, 6) if math.isfinite(atr) else np.nan,
                "stop_long": (round(long_stop, 6) if math.isfinite(long_stop) else np.nan),
                "stop_short": (round(short_stop, 6) if math.isfinite(short_stop) else np.nan),
                "vol_ann_est": (round(vol_ann, 6) if (vol_ann is not None and math.isfinite(vol_ann)) else np.nan),
                "kelly_base": BASE_KELLY,
            }
        )

    if vols_:
        diag["overall_vol_ann"] = float(np.nanmedian(vols_))

    # Write vol/stops table
    volstops = pd.DataFrame(
        out_rows,
        columns=[
            "date",
            "ticker",
            "close",
            f"atr_{ATR_LEN}",
            "stop_long",
            "stop_short",
            "vol_ann_est",
            "kelly_base",
        ],
    )
    volstops.to_csv(VOLSTOPS_CSV, index=False)

    # DD throttle map (policy)
    throttle = pd.DataFrame(
        {
            "dd_bucket_pct": [0, -5, -10, -15, -20, -30, -40],
            "risk_multiplier": [1.00, 0.95, 0.90, 0.75, 0.60, 0.40, 0.25],
            "notes": [
                "Green zone",
                "Trim a little",
                "Start throttling",
                "Throttle harder",
                "Defensive",
                "Strong defensive",
                "Capital preserve mode",
            ],
        }
    )
    throttle.to_csv(THROTTLE_CSV, index=False)

    # kill-switch YAML (write text safely)
    ks_text = f"""# reports/kill_switch.yaml
as_of: {last_day}
kelly_base: {BASE_KELLY}
target_vol_ann: {TARGET_VOL_ANN}
drawdown_limits:
  portfolio_dd_hard_pct: -0.30
  portfolio_dd_soft_pct: -0.20
  daily_pnl_soft_pct: -0.03
  daily_pnl_hard_pct: -0.05
turnover_limits:
  daily_turnover_pct_soft: 2.0
  daily_turnover_pct_hard: 3.0
capacity_limits:
  per_name_adv_pct: 12.5
  sector_gross_cap_pct: 0.45
throttle_map_csv: {THROTTLE_CSV.as_posix()}
breach_actions:
  - name: "soft-breach"
    actions: ["reduce_risk_20pct", "block_new_positions_if_gate_fail"]
  - name: "hard-breach"
    actions: ["set_risk_to_min", "notify", "freeze_rebalance"]
files:
  volstops_csv: {VOLSTOPS_CSV.as_posix()}
"""
    KILL_SWITCH_YML.write_text(ks_text, encoding="utf-8")

    diag["with_data"] = int((~volstops[f"atr_{ATR_LEN}"].isna()).sum())
    diag["rows"] = int(volstops.shape[0])

    DIAG_JSON.write_text(json.dumps(diag, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "volstops_csv": str(VOLSTOPS_CSV),
                "dd_throttle_csv": str(THROTTLE_CSV),
                "kill_switch_yaml": str(KILL_SWITCH_YML),
                "rows": int(volstops.shape[0]),
                "tickers_input": len(tickers),
                "with_atr": int((~volstops[f"atr_{ATR_LEN}"].isna()).sum()),
                "overall_vol_ann_est": diag["overall_vol_ann"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

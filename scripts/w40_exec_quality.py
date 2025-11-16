from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"F:\Projects\trading_stack_py")
REPORTS = ROOT / "reports"
PRICES = ROOT / "data" / "prices"

ORDERS = REPORTS / "wires" / "w19_orders_wire.csv"
ORDERS_FALLBACK = REPORTS / "wk12_orders_lastday.csv"

OUT_CSV = REPORTS / "wk40_exec_quality.csv"
OUT_SUMMARY = REPORTS / "wk40_exec_quality_summary.json"

# --- tunables
POV_BENCH = [0.05, 0.10, 0.15]
BASE_SPREAD_BPS = 10.0
IMPACT_K = 35.0
VWAP_NOISE_BPS = 4.0
TWAP_NOISE_BPS = 6.0
POV_NOISE_BPS = 5.0
FALLBACK_ADV_INR = 5_000_000.0


# ------------------------------ utils ------------------------------
def _pick(cols: list[str], options: list[str]) -> str | None:
    low2orig = {c.lower(): c for c in cols}
    for opt in options:
        if opt in low2orig:
            return low2orig[opt]
    return None


def _safe_num(x, default=0.0) -> float:
    try:
        v = float(x)
        return default if math.isnan(v) or math.isinf(v) else v
    except Exception:
        return default


def _load_orders() -> pd.DataFrame:
    cand = (
        ORDERS
        if ORDERS.exists()
        else (ORDERS_FALLBACK if ORDERS_FALLBACK.exists() else None)
    )
    if cand is None:
        return pd.DataFrame(
            columns=["ticker", "side", "qty", "px_ref", "notional_ref", "trade_inr"]
        )

    raw = pd.read_csv(cand)
    cols_lc = [c.lower() for c in raw.columns]

    t_col = _pick(cols_lc, ["ticker", "symbol", "instrument", "security", "name"])
    q_col = _pick(cols_lc, ["qty", "quantity", "shares", "size", "order_qty"])
    s_col = _pick(cols_lc, ["side", "side_sign", "direction", "buy_sell"])
    p_col = _pick(cols_lc, ["px_ref", "price", "px", "limit_price", "ref_price"])
    n_col = _pick(
        cols_lc, ["notional_ref", "notional", "order_notional", "value_inr", "amount"]
    )

    if not t_col or (not q_col and not n_col and not p_col):
        info = {
            "error": "orders column resolution failed",
            "orders_path": str(cand),
            "columns": list(raw.columns),
            "need_any": {
                "ticker": ["ticker", "symbol", "instrument", "security", "name"],
                "qty or notional or price": ["qty|notional|price at least one present"],
            },
        }
        OUT_SUMMARY.write_text(json.dumps(info, indent=2), encoding="utf-8")
        print(json.dumps(info, indent=2))
        return pd.DataFrame(
            columns=["ticker", "side", "qty", "px_ref", "notional_ref", "trade_inr"]
        )

    lo = {c.lower(): c for c in raw.columns}

    def oc(x: str | None) -> str | None:
        return lo.get(x) if x else None

    df = pd.DataFrame(
        {
            "ticker": raw[oc(t_col)].astype(str).str.strip(),
            "qty": pd.to_numeric(raw[oc(q_col)], errors="coerce") if q_col else np.nan,
            "side": raw[oc(s_col)] if s_col else 1,
            "px_ref": (
                pd.to_numeric(raw[oc(p_col)], errors="coerce") if p_col else np.nan
            ),
            "notional_ref": (
                pd.to_numeric(raw[oc(n_col)], errors="coerce") if n_col else np.nan
            ),
        }
    )

    # Normalize side to +1/-1 if possible
    try:
        df["side"] = pd.to_numeric(df["side"], errors="coerce").fillna(1.0)
        df.loc[df["side"] == 0, "side"] = 1.0
    except Exception:
        df["side"] = 1.0

    # Compute clean trade notionals: prefer explicit notional; else qty*px; treat NaN as missing
    n = df["notional_ref"].apply(_safe_num, default=np.nan)
    q = df["qty"].apply(_safe_num, default=np.nan)
    p = df["px_ref"].apply(_safe_num, default=np.nan)
    trade_from_prod = q * p
    trade_from_prod = trade_from_prod.where(
        ~trade_from_prod.isna() & (trade_from_prod > 0)
    )
    trade = n.where(~pd.isna(n) & (n > 0), other=trade_from_prod)
    df["trade_inr"] = trade.fillna(0.0)

    # Filter rows: need ticker, and some notional signal (either trade_inr>0 or we’ll still allow 0 but keep ticker)
    df["ticker"] = df["ticker"].replace({"": np.nan})
    df = df.dropna(subset=["ticker"]).reset_index(drop=True)

    return df


def _adv_from_prices(ticker: str) -> float | None:
    p = PRICES / f"{ticker}.parquet"
    if not p.exists():
        return None
    try:
        x = pd.read_parquet(p)
        cols = {c.lower(): c for c in x.columns}
        c = cols.get("close") or cols.get("px_close") or cols.get("price")
        v = cols.get("volume")
        if not c or not v:
            return None
        s = x[[c, v]].dropna().copy()
        s["turnover_inr"] = pd.to_numeric(s[c], errors="coerce") * pd.to_numeric(
            s[v], errors="coerce"
        )
        s = s.dropna()
        return float(s["turnover_inr"].tail(60).mean()) if not s.empty else None
    except Exception:
        return None


def _slippage_bps_for_notional(trade_inr: float, adv_inr: float) -> float:
    adv = adv_inr if adv_inr and adv_inr > 0 else FALLBACK_ADV_INR
    trade = max(trade_inr, 0.0)
    pct_adv = trade / adv
    impact = IMPACT_K * np.sqrt(pct_adv)
    return float(BASE_SPREAD_BPS + impact)


def _simulate_styles(row: pd.Series) -> list[dict]:
    t = str(row["ticker"])
    notional = _safe_num(row.get("trade_inr"), default=0.0)
    if notional <= 0:
        # if qty & px exist but trade_inr didn’t compute (edge coercion), try again
        q = _safe_num(row.get("qty"), default=0.0)
        px = _safe_num(row.get("px_ref"), default=0.0)
        notional = q * px

    adv = _adv_from_prices(t) or FALLBACK_ADV_INR

    vwap_bps = _slippage_bps_for_notional(notional, adv) + VWAP_NOISE_BPS
    twap_bps = _slippage_bps_for_notional(notional, adv) + TWAP_NOISE_BPS

    pov_rows = []
    for pov in POV_BENCH:
        pov_bps = _slippage_bps_for_notional(notional, adv) + POV_NOISE_BPS
        pov_rows.append(("POV", pov, pov_bps))
    pov_best = min(pov_rows, key=lambda x: x[2])

    return [
        {
            "ticker": t,
            "style": "VWAP",
            "param": "n/a",
            "slippage_bps": round(vwap_bps, 3),
            "adv_inr": adv,
            "trade_inr": notional,
        },
        {
            "ticker": t,
            "style": "TWAP",
            "param": "n/a",
            "slippage_bps": round(twap_bps, 3),
            "adv_inr": adv,
            "trade_inr": notional,
        },
        {
            "ticker": t,
            "style": pov_best[0],
            "param": f"pov={int(pov_best[1] * 100)}%",
            "slippage_bps": round(pov_best[2], 3),
            "adv_inr": adv,
            "trade_inr": notional,
        },
    ]


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    orders = _load_orders()
    if orders.empty:
        return

    rows = []
    for _, r in orders.iterrows():
        rows += _simulate_styles(r)

    df = pd.DataFrame(rows)

    if df.empty:
        info = {
            "error": "No rows after simulation",
            "orders_rows": int(orders.shape[0]),
        }
        OUT_SUMMARY.write_text(json.dumps(info, indent=2), encoding="utf-8")
        print(json.dumps(info, indent=2))
        return

    # Guard against any NaNs in slippage due to unexpected inputs
    df["slippage_bps"] = pd.to_numeric(df["slippage_bps"], errors="coerce")
    df = df.dropna(subset=["ticker", "slippage_bps"])

    df["rank"] = df.groupby("ticker")["slippage_bps"].rank(method="first")
    best = df.loc[df["rank"] == 1.0].copy().sort_values(["slippage_bps", "ticker"])

    style_means = df.groupby("style")["slippage_bps"].mean().to_dict()
    port_best_mean = (
        float(best["slippage_bps"].mean()) if not best.empty else float("nan")
    )

    df.sort_values(["ticker", "slippage_bps"], inplace=True)
    df.to_csv(OUT_CSV, index=False)

    summary = {
        "as_of_ist": datetime.now().astimezone().isoformat(),
        "orders": int(orders.shape[0]),
        "tickers": int(best["ticker"].nunique()),
        "portfolio_best_mean_bps": (
            None if math.isnan(port_best_mean) else round(port_best_mean, 3)
        ),
        "style_means_bps": {
            k: (None if (v is None or math.isnan(float(v))) else round(float(v), 3))
            for k, v in style_means.items()
        },
        "files": {"detail_csv": str(OUT_CSV)},
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

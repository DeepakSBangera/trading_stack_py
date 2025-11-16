from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PRICES = DATA / "prices"
CSV = DATA / "csv"
REPORTS = ROOT / "reports"

# ---------------------------- Config ----------------------------
PORTFOLIO_NOTIONAL_INR = float(os.getenv("W39_NOTIONAL_INR", "10000000"))  # ₹1.0 crore
MIN_ADV_INR = float(os.getenv("W39_MIN_ADV_INR", "5000000"))  # ₹50 lakhs
MAX_ADV_PCT = float(os.getenv("W39_MAX_ADV_PCT", "0.10"))  # 10% of ADV
ADV_LOOKBACK = int(os.getenv("W39_ADV_LOOKBACK", "20"))

WEIGHT_SOURCES = [
    REPORTS / "wk43_barbell_compare.csv",
    REPORTS / "wk41_momentum_tilt.csv",
    REPORTS / "wk11_alpha_blend.csv",
    REPORTS / "wk12_kelly_dd.csv",
]

DETAIL_CSV = REPORTS / "wk39_capacity_audit.csv"
SUMMARY_JSON = REPORTS / "wk39_capacity_summary.json"


def _pick(cols: list[str], want: list[str]) -> str | None:
    m = {c.lower(): c for c in cols}
    for w in want:
        if w.lower() in m:
            return m[w.lower()]
    return None


def _load_weights() -> pd.DataFrame:
    latest: pd.DataFrame | None = None
    src_used: str | None = None

    for path in WEIGHT_SOURCES:
        if path.exists():
            try:
                x = pd.read_csv(path, parse_dates=["date"])
                t = _pick(list(x.columns), ["ticker", "symbol", "name"])
                w = _pick(
                    list(x.columns), ["w", "weight", "w_total", "w_capped", "w_norm"]
                )
                d = _pick(list(x.columns), ["date", "dt"])
                if t is None or w is None:
                    continue
                if d is None and "date" not in x.columns:
                    x["date"] = pd.Timestamp("today").normalize()
                    d = "date"
                x = (
                    x[[d, t, w]]
                    .rename(columns={d: "date", t: "ticker", w: "w"})
                    .dropna()
                )
                last = x["date"].max()
                x = x[x["date"] == last].copy()
                x["w"] = x["w"].astype(float)
                s = float(x["w"].abs().sum())
                if s > 0:
                    x["w"] = x["w"] / s
                latest = x[["ticker", "w"]].copy()
                src_used = str(path)
                break
            except Exception:
                pass

    if latest is not None and not latest.empty:
        latest["ticker"] = latest["ticker"].astype(str)
        return latest.assign(source=src_used if src_used else "unknown")

    # Fallback: derive tickers from price files and make equal weights (top 30)
    tickers = set()
    if PRICES.exists():
        for p in PRICES.glob("*.parquet"):
            tickers.add(p.stem.split(".")[0])
    if not tickers and CSV.exists():
        for p in CSV.glob("*.csv"):
            tickers.add(p.stem.split(".")[0])

    tickers = sorted(list(tickers))[:30]
    if not tickers:
        raise SystemExit("No weights and no tickers found to build capacity audit.")

    w = pd.DataFrame({"ticker": tickers, "w": 1.0 / max(1, len(tickers))})
    return w.assign(source="synthetic_equal")


def _load_panel_close_vol() -> pd.DataFrame:
    """Return [date, ticker, close, volume] panel from Parquet/CSV."""
    frames = []

    def _from_parquet(path: Path) -> pd.DataFrame | None:
        try:
            x = pd.read_parquet(path)
            cols = list(x.columns)
            d = _pick(cols, ["date", "dt"])
            c = _pick(cols, ["close", "px_close", "price"])
            v = _pick(cols, ["volume", "qty", "shares"])
            if d is None or c is None:
                return None
            out = x[[d, c] + ([v] if v else [])].rename(columns={d: "date", c: "close"})
            out["volume"] = out[v] if v else np.nan
            out = out.drop(columns=[v], errors="ignore")
            out["ticker"] = path.stem.split(".")[0]
            return out
        except Exception:
            return None

    def _from_csv(path: Path) -> pd.DataFrame | None:
        try:
            x = pd.read_csv(path, parse_dates=["date"], infer_datetime_format=True)
            cols = list(x.columns)
            d = _pick(cols, ["date", "dt"])
            c = _pick(cols, ["close", "px_close", "price", "adj_close"])
            v = _pick(cols, ["volume", "qty", "shares"])
            if d is None or c is None:
                return None
            out = x[[d, c] + ([v] if v else [])].rename(columns={d: "date", c: "close"})
            out["volume"] = out[v] if v else np.nan
            out = out.drop(columns=[v], errors="ignore")
            out["ticker"] = path.stem.split(".")[0]
            return out
        except Exception:
            return None

    if PRICES.exists():
        for p in PRICES.glob("*.parquet"):
            d = _from_parquet(p)
            if d is not None:
                frames.append(d)
    if not frames and CSV.exists():
        for p in CSV.glob("*.csv"):
            d = _from_csv(p)
            if d is not None:
                frames.append(d)

    if not frames:
        raise SystemExit(
            "No price files found in data/prices or data/csv with close/volume."
        )

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.dropna(subset=["close"])
    panel["date"] = pd.to_datetime(panel["date"]).dt.tz_localize(None)

    # ✅ FIX: use transform to keep index aligned; then fill remaining with 0
    panel["volume"] = (
        panel.groupby("ticker", sort=False)["volume"]
        .transform(lambda s: s.fillna(s.median()))
        .fillna(0)
        .astype(float)
    )
    return panel.sort_values(["ticker", "date"]).reset_index(drop=True)


def _adv_inr(panel: pd.DataFrame, lookback: int) -> pd.Series:
    """20d ADV in INR ≈ mean(close×volume) over trailing window, per ticker."""
    df = panel.sort_values(["ticker", "date"]).copy()
    df["notional"] = (df["close"].astype(float) * df["volume"].astype(float)).astype(
        float
    )

    # return one scalar per group (not multiindex)
    def last_roll_mean(s: pd.Series) -> float:
        if s.empty:
            return np.nan
        roll = s.rolling(lookback, min_periods=max(5, lookback // 2)).mean()
        return float(roll.iloc[-1])

    adv = df.groupby("ticker", sort=False)["notional"].apply(last_roll_mean)
    return adv.fillna(0.0)


def _capacity_table(weights: pd.DataFrame, adv_map: pd.Series) -> pd.DataFrame:
    w = weights.copy()
    w["ticker"] = w["ticker"].astype(str)
    w = w.groupby("ticker", as_index=False)["w"].sum()
    s = float(w["w"].abs().sum())
    if s > 0:
        w["w"] = w["w"] / s

    adv_df = adv_map.rename("adv_inr").reset_index()
    adv_df.columns = ["ticker", "adv_inr"]
    out = w.merge(adv_df, on="ticker", how="left").fillna({"adv_inr": 0.0})

    out["notional_inr"] = out["w"].abs() * PORTFOLIO_NOTIONAL_INR
    out["adv_pct_baseline"] = np.where(
        out["adv_inr"] > 0, out["notional_inr"] / out["adv_inr"], np.inf
    )
    for mult in (2, 3):
        out[f"adv_pct_{mult}x"] = out["adv_pct_baseline"] * mult

    out["viol_min_adv"] = out["adv_inr"] < MIN_ADV_INR
    out["viol_cap_baseline"] = out["adv_pct_baseline"] > MAX_ADV_PCT
    out["viol_cap_2x"] = out["adv_pct_2x"] > MAX_ADV_PCT
    out["viol_cap_3x"] = out["adv_pct_3x"] > MAX_ADV_PCT

    def reco_row(r) -> str:
        hints = []
        if r["viol_min_adv"]:
            hints.append("min-ADV")
        if r["viol_cap_baseline"]:
            hints.append("cap@1x")
        if r["viol_cap_2x"]:
            hints.append("cap@2x")
        if r["viol_cap_3x"]:
            hints.append("cap@3x")
        return ",".join(hints) if hints else "ok"

    out["reco"] = out.apply(reco_row, axis=1)
    cols = [
        "ticker",
        "w",
        "adv_inr",
        "notional_inr",
        "adv_pct_baseline",
        "adv_pct_2x",
        "adv_pct_3x",
        "viol_min_adv",
        "viol_cap_baseline",
        "viol_cap_2x",
        "viol_cap_3x",
        "reco",
    ]
    out = out[cols].sort_values(["reco", "adv_pct_baseline"], ascending=[True, False])
    return out


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    weights = _load_weights()
    panel = _load_panel_close_vol()
    adv = _adv_inr(panel, ADV_LOOKBACK)
    tab = _capacity_table(weights, adv)

    n = int(tab.shape[0])
    viol_min = int(tab["viol_min_adv"].sum())
    viol_1x = int(tab["viol_cap_baseline"].sum())
    viol_2x = int(tab["viol_cap_2x"].sum())
    viol_3x = int(tab["viol_cap_3x"].sum())

    detail = tab.copy()
    for c in ["adv_inr", "notional_inr"]:
        detail[c] = detail[c].round(2)
    for c in ["adv_pct_baseline", "adv_pct_2x", "adv_pct_3x"]:
        detail[c] = (detail[c] * 100).replace(np.inf, np.nan).round(3)

    detail.to_csv(DETAIL_CSV, index=False)

    summary = {
        "as_of_ist": pd.Timestamp.now(tz="Asia/Kolkata").isoformat(),
        "names": n,
        "portfolio_notional_inr": PORTFOLIO_NOTIONAL_INR,
        "min_adv_inr": MIN_ADV_INR,
        "max_adv_pct": MAX_ADV_PCT,
        "adv_lookback_days": ADV_LOOKBACK,
        "violations": {
            "min_adv": viol_min,
            "cap_1x": viol_1x,
            "cap_2x": viol_2x,
            "cap_3x": viol_3x,
        },
        "files": {"detail_csv": str(DETAIL_CSV)},
        "notes": "ADV ≈ mean(close×volume, last 20d). adv_pct columns are % of a single day's ADV; lower is safer.",
    }
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

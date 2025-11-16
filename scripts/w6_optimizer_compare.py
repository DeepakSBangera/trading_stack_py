# scripts/w6_optimizer_compare.py
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"F:\Projects\trading_stack_py")
REPORTS = ROOT / "reports"
DATA = ROOT / "data" / "prices"
META = ROOT / "data" / "meta"
REPORTS.mkdir(parents=True, exist_ok=True)

TARGETS_CSV = REPORTS / "wk11_blend_targets.csv"  # input
ADV_PARQUET = REPORTS / "adv_value.parquet"  # optional ADV in INR (ticker, adv_value)
SECTOR_MAP_CSV = META / "sector_map.csv"  # optional mapping: ticker,sector

OUT_COMPARE = REPORTS / "wk6_portfolio_compare.csv"
OUT_EXPO = REPORTS / "factor_exposure_weekly.csv"
OUT_CAPACITY = REPORTS / "capacity_curve.csv"
OUT_WE_CAP = (
    REPORTS / "wk6_weights_capped.csv"
)  # produced by a helper here (soft caps demo)
OUT_DIAG = REPORTS / "w6_diag.json"

# --- knobs ---
VOL_LOOKBACK_D = 20
DEFAULT_DAILY_VOL = 0.20 / math.sqrt(252)  # fallback
CAP_PER_NAME_ADV_PCT = 12.5  # per-name ADV cap for capacity curve
SECTOR_CAP = 0.45  # 45% max sector gross
NAME_CAP = 0.10  # 10% max per name gross (soft; for demo)
CAPACITY_NOTIONALS = [
    1_000_000,
    2_500_000,
    5_000_000,
    10_000_000,
    20_000_000,
    50_000_000,
]


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


def _load_targets():
    if not TARGETS_CSV.exists():
        raise SystemExit(f"Missing {TARGETS_CSV}. Run W11 first.")
    df = pd.read_csv(TARGETS_CSV)
    dcol = _pick(df.columns, ["date", "dt", "trading_day", "asof", "as_of"])
    tcol = _pick(df.columns, ["ticker", "symbol", "name"])
    wcol = _pick(
        df.columns, ["target_w", "weight", "w", "blend_w", "final_w", "target_weight"]
    )
    if not dcol or not tcol or not wcol:
        raise SystemExit("wk11_blend_targets.csv missing date/ticker/weight columns.")
    df[dcol] = pd.to_datetime(df[dcol], errors="coerce").dt.date
    df = wdf = df[[dcol, tcol, wcol]].rename(
        columns={dcol: "date", tcol: "ticker", wcol: "w"}
    )
    return wdf


def _load_prices(ticker: str) -> pd.DataFrame | None:
    p = DATA / f"{ticker}.parquet"
    if not p.exists():
        return None
    try:
        x = pd.read_parquet(p)
        cols = {c.lower(): c for c in x.columns}
        d = _pick(x.columns, ["date", "dt"])
        c = _pick(x.columns, ["close", "px_close", "price"])
        if not d or not c:
            return None
        out = x[[d, c]].copy()
        out.columns = ["date", "close"]
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out = out.dropna().sort_values("date")
        return out
    except Exception:
        return None


def _daily_vol(tic: str, upto_date: pd.Timestamp) -> float:
    df = _load_prices(tic)
    if df is None:
        return DEFAULT_DAILY_VOL
    df = df[df["date"] <= pd.to_datetime(upto_date)]
    if df.shape[0] < VOL_LOOKBACK_D + 2:
        return DEFAULT_DAILY_VOL
    r = df["close"].pct_change().dropna().tail(VOL_LOOKBACK_D)
    if r.empty:
        return DEFAULT_DAILY_VOL
    v = float(r.std(ddof=0))
    return v if math.isfinite(v) and v > 0 else DEFAULT_DAILY_VOL


def _equal_weight(tickers: list[str]) -> np.ndarray:
    n = len(tickers)
    if n == 0:
        return np.array([])
    return np.ones(n) / n


def _inv_vol_weights(tickers: list[str], asof) -> np.ndarray:
    vols = np.array([_daily_vol(t, asof) for t in tickers])
    inv = np.where(vols > 0, 1.0 / vols, 0.0)
    if inv.sum() == 0:
        return _equal_weight(tickers)
    return inv / inv.sum()


def _mv_shrink_weights(tickers: list[str], asof, w_prior: np.ndarray) -> np.ndarray:
    # tiny demo: convex mix between inv-vol and prior target (like shrinkage)
    inv = _inv_vol_weights(tickers, asof)
    alpha = 0.5
    w = alpha * inv + (1 - alpha) * w_prior
    s = w.sum()
    return w / s if s > 0 else _equal_weight(tickers)


def _herfindahl(w: np.ndarray) -> float:
    return float((w**2).sum())


def _gini(w: np.ndarray) -> float:
    w = np.sort(np.abs(w))
    n = len(w)
    if n == 0 or w.sum() == 0:
        return 0.0
    cum = np.cumsum(w)
    g = 1 - (2 / (n - 1)) * (n - cum.sum() / (w.sum()))
    # quick stable formula
    i = np.arange(1, n + 1)
    g = 1 - (2 / (n - 1)) * ((n + 1 - (2 * (i * w).sum() / w.sum())) / 2)
    return float(max(0, min(1, g)))


def _load_adv_map() -> dict[str, float]:
    if not ADV_PARQUET.exists():
        return {}
    try:
        adv = pd.read_parquet(ADV_PARQUET)
        cols = {c.lower(): c for c in adv.columns}
        t = cols.get("ticker")
        a = cols.get("adv_value")
        if t and a:
            return dict(
                zip(
                    adv[t].astype(str),
                    pd.to_numeric(adv[a], errors="coerce").fillna(0.0),
                    strict=False,
                )
            )
    except Exception:
        pass
    return {}


def _load_sector_map() -> dict[str, str]:
    if not SECTOR_MAP_CSV.exists():
        return {}
    try:
        m = pd.read_csv(SECTOR_MAP_CSV)
        cols = {c.lower(): c for c in m.columns}
        t = _pick(m.columns, ["ticker", "symbol", "name"])
        s = _pick(m.columns, ["sector", "industry", "group"])
        if t and s:
            return {str(r[t]): str(r[s]) for _, r in m.iterrows()}
    except Exception:
        pass
    return {}


def _capacity_curve(df: pd.DataFrame, scheme: str, adv_map: dict) -> list[dict]:
    rows = []
    # df: columns date,ticker,w_<scheme>
    for N in CAPACITY_NOTIONALS:
        for d, block in df.groupby("date"):
            need = 0.0
            names = 0
            for _, r in block.iterrows():
                w = float(r[f"w_{scheme}"])
                if w == 0:
                    continue
                names += 1
                adv = adv_map.get(str(r["ticker"]), 5e7)  # fallback INR 5e7
                need += min(
                    abs(w) * N / adv * 100.0, CAP_PER_NAME_ADV_PCT
                )  # sum of per-name utilisation clipped
            rows.append(
                {
                    "date": d,
                    "scheme": scheme,
                    "notional_inr": N,
                    "avg_adv_util_pct": (need / max(1, names)),
                }
            )
    return rows


def _enforce_caps(weights: pd.DataFrame, sector_map: dict) -> pd.DataFrame:
    # soft enforcement: clip per name to NAME_CAP; then sector gross to SECTOR_CAP; renormalize to 1.0
    out = []
    for d, g in weights.groupby("date"):
        tmp = g.copy()
        # clip
        tmp["w_capped"] = tmp["w_mvshrink"].clip(-NAME_CAP, NAME_CAP)
        # sector cap
        tmp["sector"] = [
            sector_map.get(t, "UNKNOWN") for t in tmp["ticker"].astype(str)
        ]
        # scale down sectors above cap proportionally
        sector_sum = tmp.groupby("sector")["w_capped"].apply(lambda x: x.abs().sum())
        scale = {
            s: (SECTOR_CAP / val) if val > SECTOR_CAP else 1.0
            for s, val in sector_sum.items()
        }
        tmp["w_capped"] = tmp.apply(
            lambda r: r["w_capped"] * scale.get(r["sector"], 1.0), axis=1
        )
        # final renorm to sum(|w|)=1 (gross normalize)
        gross = tmp["w_capped"].abs().sum()
        if gross > 0:
            tmp["w_capped"] = tmp["w_capped"] / gross
        out.append(tmp[["date", "ticker", "sector", "w_capped"]])
    return pd.concat(out, ignore_index=True)


def main():
    diag = {"notes": []}
    wdf = _load_targets()
    last = wdf["date"].max()
    # build universe last day (use last day composition)
    univ = wdf[wdf["date"] == last].copy()
    tickers = univ["ticker"].astype(str).tolist()
    prior = univ["w"].to_numpy(dtype=float)
    prior = prior / abs(prior).sum() if abs(prior).sum() > 0 else _equal_weight(tickers)

    ew = _equal_weight(tickers)
    iv = _inv_vol_weights(tickers, last)
    mv = _mv_shrink_weights(tickers, last, prior)

    res = []
    for nm, arr in [("ew", ew), ("invvol", iv), ("mvshrink", mv)]:
        res.append(
            {
                "date": last,
                "scheme": nm,
                "n": len(arr),
                "sum_w_abs": float(np.abs(arr).sum()),
                "max_w": float(np.max(np.abs(arr))) if len(arr) else 0.0,
                "herfindahl": _herfindahl(arr),
                "gini_approx": _gini(arr),
            }
        )
    compare = pd.DataFrame(res)
    compare.to_csv(OUT_COMPARE, index=False)

    # per-ticker table of weights for exposures & capacity
    weights = pd.DataFrame(
        {"date": last, "ticker": tickers, "w_ew": ew, "w_invvol": iv, "w_mvshrink": mv}
    )

    # exposures by sector (if map present)
    sector_map = _load_sector_map()
    weights["sector"] = [
        sector_map.get(t, "UNKNOWN") for t in weights["ticker"].astype(str)
    ]
    expo = (
        weights.melt(
            id_vars=["date", "ticker", "sector"],
            value_vars=["w_ew", "w_invvol", "w_mvshrink"],
            var_name="scheme",
            value_name="w",
        )
        .groupby(["date", "sector", "scheme"], as_index=False)["w"]
        .sum()
    )
    expo.to_csv(OUT_EXPO, index=False)

    # capacity curve (ADV map optional)
    adv_map = _load_adv_map()
    cap_rows = []
    for scheme in ["ew", "invvol", "mvshrink"]:
        cap_rows += _capacity_curve(
            weights.rename(columns={f"w_{scheme}": f"w_{scheme}"}), scheme, adv_map
        )
    cap = pd.DataFrame(cap_rows)
    cap.to_csv(OUT_CAPACITY, index=False)

    # enforce caps (name + sector) on mvshrink as the working scheme
    capped = _enforce_caps(weights, sector_map)
    capped.to_csv(OUT_WE_CAP, index=False)

    OUT_DIAG.write_text(
        json.dumps(
            {
                "as_of": str(last),
                "universe": len(tickers),
                "sector_map_present": bool(sector_map),
                "adv_map_present": bool(adv_map),
                "outputs": [
                    str(OUT_COMPARE),
                    str(OUT_EXPO),
                    str(OUT_CAPACITY),
                    str(OUT_WE_CAP),
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "compare_csv": str(OUT_COMPARE),
                "exposure_csv": str(OUT_EXPO),
                "capacity_csv": str(OUT_CAPACITY),
                "weights_capped_csv": str(OUT_WE_CAP),
                "universe": len(tickers),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

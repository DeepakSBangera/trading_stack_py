from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# --------- Paths / constants ----------
ROOT = Path(r"F:\Projects\trading_stack_py")
REPORTS = ROOT / "reports"
DATA = ROOT / "data" / "csv"
OUT_CSV = REPORTS / "wk42_after_tax_schedule.csv"
OUT_SUMMARY = REPORTS / "wk42_after_tax_schedule_summary.json"

# optional inputs we try in this order
LOT_CANDIDATES = [
    REPORTS / "lots_history.csv",  # preferred if you already export lots
    DATA / "lots_history.csv",  # seed/synthetic
    REPORTS / "positions_lots.csv",
    DATA / "positions_lots.csv",
]

DIV_CANDIDATES = [
    DATA / "dividends.csv",  # columns: ticker, ex_date (or ex-dividend), amount (optional)
    REPORTS / "dividends.csv",
]

# India rules (simplified): LTCG >= 365 days; STCG < 365
LTCG_DAYS = 365


def _pick(cols: list[str], want: list[str]) -> str | None:
    low = {c.lower(): c for c in cols}
    for w in want:
        if w.lower() in low:
            return low[w.lower()]
    return None


def _load_first_existing(cands: list[Path]) -> pd.DataFrame | None:
    for p in cands:
        if p.exists():
            if p.suffix.lower() == ".csv":
                return pd.read_csv(p)
            elif p.suffix.lower() == ".parquet":
                return pd.read_parquet(p)
    return None


def _normalize_lots(df: pd.DataFrame) -> pd.DataFrame:
    """Make a clean lots table with: acquired_date, ticker, qty, cost, mkt_px(optional)."""
    cols = df.columns.tolist()
    dt = _pick(cols, ["acquired_date", "buy_date", "open_date", "lot_date", "date"])
    tk = _pick(cols, ["ticker", "symbol", "name"])
    q = _pick(cols, ["qty", "quantity", "shares"])
    c = _pick(cols, ["cost", "cost_price", "buy_px", "avg_cost"])
    px = _pick(cols, ["mkt_px", "mark_price", "close", "price", "last"])

    if any(x is None for x in [dt, tk, q, c]):
        raise SystemExit("Lots file missing required columns (need date/ticker/qty/cost).")

    out = df[[dt, tk, q, c] + ([px] if px else [])].copy()
    out.rename(
        columns={
            dt: "acquired_date",
            tk: "ticker",
            q: "qty",
            c: "cost",
            (px or "mkt_px"): "mkt_px",
        },
        inplace=True,
    )
    out["acquired_date"] = pd.to_datetime(out["acquired_date"])
    out["qty"] = pd.to_numeric(out["qty"], errors="coerce").fillna(0.0)
    out["cost"] = pd.to_numeric(out["cost"], errors="coerce")
    if "mkt_px" in out.columns:
        out["mkt_px"] = pd.to_numeric(out["mkt_px"], errors="coerce")
    return out.dropna(subset=["acquired_date", "ticker"]).reset_index(drop=True)


def _normalize_divs(df: pd.DataFrame) -> pd.DataFrame:
    cols = df.columns.tolist()
    tk = _pick(cols, ["ticker", "symbol", "name"])
    dx = _pick(cols, ["ex_date", "ex-dividend", "exdiv", "date"])
    if tk is None or dx is None:
        raise SystemExit("Dividends file missing columns (need ticker + ex_date).")
    out = df[[tk, dx]].copy().rename(columns={tk: "ticker", dx: "ex_date"})
    out["ex_date"] = pd.to_datetime(out["ex_date"])
    return out.dropna().reset_index(drop=True)


def _best_exdate_after(ticker: str, from_date: pd.Timestamp, divs: pd.DataFrame | None) -> pd.Timestamp | None:
    if divs is None:
        return None
    d = divs[(divs["ticker"].astype(str) == str(ticker)) & (divs["ex_date"] >= from_date)]
    if d.empty:
        return None
    return pd.to_datetime(d["ex_date"].min())


def _schedule(df: pd.DataFrame, divs: pd.DataFrame | None, today_ist: pd.Timestamp) -> pd.DataFrame:
    df = df.copy()
    df["today"] = today_ist.normalize()
    df["holding_days"] = (df["today"] - df["acquired_date"]).dt.days
    df["tax_band"] = np.where(df["holding_days"] >= LTCG_DAYS, "LTCG", "STCG")

    # when will this lot flip to LTCG?
    need = (LTCG_DAYS - df["holding_days"]).clip(lower=0)
    df["days_to_ltcg"] = need
    df["ltcg_date"] = df["today"] + pd.to_timedelta(need, unit="D")

    # recommended sell date:
    # - if already LTCG: today (no tax band penalty)
    # - else: ltcg_date (or next day after closest ex-date window to avoid dividend-dilution if desired)
    rec = []
    for _, r in df.iterrows():
        if r["tax_band"] == "LTCG":
            base = r["today"]
        else:
            base = r["ltcg_date"]
        # optional nudge: avoid selling just before ex-date (T+1 slippage). If ex-date within +3 days of base, push +3.
        exn = _best_exdate_after(str(r["ticker"]), base, divs)
        if exn is not None and (exn - base).days <= 3:
            base = exn + pd.Timedelta(days=1)
        rec.append(base)
    df["recommended_sell_date"] = pd.to_datetime(rec)

    # per-lot economics (if market price available)
    if "mkt_px" in df.columns:
        df["unrealized_pnl"] = (df["mkt_px"] - df["cost"]) * df["qty"]
    else:
        df["unrealized_pnl"] = np.nan

    # compact output
    out = df[
        [
            "ticker",
            "acquired_date",
            "qty",
            "cost",
            "mkt_px",
            "holding_days",
            "tax_band",
            "days_to_ltcg",
            "ltcg_date",
            "recommended_sell_date",
            "unrealized_pnl",
        ]
    ].sort_values(["tax_band", "days_to_ltcg", "ticker"], ascending=[True, True, True])

    return out.reset_index(drop=True)


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)

    lots_raw = _load_first_existing(LOT_CANDIDATES)
    if lots_raw is None:
        # Create a tiny synthetic example so the report still renders
        today = pd.Timestamp.utcnow().tz_convert("Asia/Kolkata").normalize()
        syn = pd.DataFrame(
            {
                "acquired_date": [
                    today - pd.Timedelta(days=100),
                    today - pd.Timedelta(days=420),
                ],
                "ticker": ["SYNTH1.NS", "SYNTH2.NS"],
                "qty": [10, 20],
                "cost": [100.0, 200.0],
                "mkt_px": [118.0, 245.0],
            }
        )
        lots = _normalize_lots(syn)
        seeded = True
    else:
        lots = _normalize_lots(lots_raw)
        seeded = False

    divs_raw = _load_first_existing(DIV_CANDIDATES)
    divs = _normalize_divs(divs_raw) if divs_raw is not None else None

    today_ist = pd.Timestamp.utcnow().tz_convert("Asia/Kolkata")
    out = _schedule(lots, divs, today_ist)

    out.to_csv(OUT_CSV, index=False)

    # roll-up stats
    band_counts = out["tax_band"].value_counts().to_dict()
    soon = out[(out["tax_band"] == "STCG") & (out["days_to_ltcg"] <= 30)].shape[0]

    summary = {
        "as_of_ist": today_ist.isoformat(),
        "lots_rows": int(out.shape[0]),
        "by_tax_band": band_counts,
        "stcg_lots_flipping_within_30d": int(soon),
        "files": {"detail_csv": str(OUT_CSV)},
        "inputs": {
            "lots_source": "synthetic_example" if seeded else "loaded_file",
            "dividends_loaded": bool(divs_raw is not None),
        },
        "notes": "LTCG threshold = 365d; recommended_sell_date nudged to avoid ex-date within +3d.",
    }
    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

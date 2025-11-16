from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"F:\Projects\trading_stack_py")
REPORTS = ROOT / "reports"
CONFIG = ROOT / "config" / "capacity_policy.yaml"

# Inputs
TARGETS = REPORTS / "wk11_blend_targets.csv"  # from W11
SCHED = REPORTS / "risk_schedule_blended.csv"  # from W8
POSPQ = (
    REPORTS / "positions_daily.parquet"
)  # current portfolio snapshot (weights, port_value)
ADV_PQ = (
    REPORTS / "adv_value.parquet"
)  # ADV in currency per name per day (if available)

# Outputs
OUT_SCHEDULE = REPORTS / "wk12_orders_schedule.csv"  # orders for all dates
OUT_TODAY = REPORTS / "wk12_orders_lastday.csv"  # last-day slice
OUT_VCHECK = REPORTS / "wk12_orders_validation.csv"  # simple checks summary


def open_win(p: Path):
    if sys.platform.startswith("win") and p.exists():
        os.startfile(p)


def load_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def latest_snapshot_weights() -> pd.Series:
    if not POSPQ.exists():
        raise SystemExit("Missing reports\\positions_daily.parquet. Bootstrap first.")
    pos = pd.read_parquet(POSPQ)
    pos["date"] = pd.to_datetime(pos["date"])
    d = pos["date"].max()
    snap = pos[pos["date"] == d]
    if {"ticker", "weight"}.issubset(snap.columns):
        w = snap.set_index("ticker")["weight"].clip(lower=0.0)
        s = float(w.sum())
        return (w / s) if s > 0 else w
    raise SystemExit("positions_daily.parquet lacks ticker/weight columns.")


def main():
    # Load targets and schedule (for blocks)
    if not TARGETS.exists():
        raise SystemExit("Missing wk11_blend_targets.csv — run W11 first.")
    t = pd.read_csv(TARGETS, parse_dates=["date"])
    sched = pd.read_csv(SCHED, parse_dates=["date"]) if SCHED.exists() else None

    # Current weights from latest snapshot
    w_now = latest_snapshot_weights()

    # Build orders per date: action = target_w - current_w (rebal allowed); if add blocked & current=0 → 0
    rows = []
    for d, df in t.groupby("date", sort=True):
        df = df.copy().set_index("ticker")
        df["target_w"] = df["target_w"].fillna(0.0).clip(lower=0.0)

        # derive allow flags (already embedded in W11 targets logic, but keep for clarity)
        allow_new = (
            df["allow_new_final"].fillna(True).astype(bool)
            if "allow_new_final" in df.columns
            else True
        )
        reb_ok = (
            df["rebalance_allowed_final"].fillna(True).astype(bool)
            if "rebalance_allowed_final" in df.columns
            else True
        )

        # baseline current for this date: use latest snapshot (simple demo)
        cur = w_now.reindex(df.index).fillna(0.0)

        delta = df["target_w"] - cur
        # if rebalancing globally blocked → freeze (delta=0)
        if isinstance(reb_ok, pd.Series) and not reb_ok.all():
            frozen = ~reb_ok
            delta.loc[frozen.index[frozen]] = 0.0

        # if allow_new=False and cur==0 → block adds
        if isinstance(allow_new, pd.Series):
            add_block = (~allow_new) & (cur <= 1e-12) & (df["target_w"] > 0)
            delta.loc[add_block.index[add_block]] = 0.0

        # classify action
        action = np.where(delta > 1e-6, "BUY", np.where(delta < -1e-6, "SELL", "HOLD"))

        rows += [
            {
                "date": d.date().isoformat(),
                "ticker": k,
                "current_w": float(cur.get(k, 0.0)),
                "target_w": float(df.at[k, "target_w"]),
                "delta_w": float(delta.get(k, 0.0)),
                "action": a,
            }
            for k, a in zip(df.index, action, strict=False)
        ]

    sched_df = pd.DataFrame(rows).sort_values(["date", "ticker"])
    REPORTS.mkdir(parents=True, exist_ok=True)
    sched_df.to_csv(OUT_SCHEDULE, index=False)

    # Last-day slice
    last_day = sched_df["date"].max()
    last = sched_df[sched_df["date"] == last_day].copy()
    last.to_csv(OUT_TODAY, index=False)

    # Validation: name/sector caps & ADV (if present)
    checks = []
    # name cap
    name_cap = 0.12
    last["name_cap_breach"] = last["target_w"] > name_cap
    checks.append(
        {
            "check": "per_name_cap_breach_any",
            "value": bool(last["name_cap_breach"].any()),
            "limit": "<=0.12",
        }
    )

    # sector cap (reconstruct sector quickly)
    SECTOR_MAP = {
        "HDFCBANK.NS": "Financials",
        "RELIANCE.NS": "Energy",
        "TCS.NS": "IT",
        "INFY.NS": "IT",
        "AXISBANK.NS": "Financials",
        "ICICIBANK.NS": "Financials",
        "SBIN.NS": "Financials",
        "BHARTIARTL.NS": "Telecom",
        "ITC.NS": "Staples",
        "LT.NS": "Industrials",
        "KOTAKBANK.NS": "Financials",
        "BAJFINANCE.NS": "Financials",
        "HCLTECH.NS": "IT",
        "MARUTI.NS": "Discretionary",
        "SUNPHARMA.NS": "Healthcare",
        "TITAN.NS": "Discretionary",
        "ULTRACEMCO.NS": "Materials",
        "ONGC.NS": "Energy",
        "NESTLEIND.NS": "Staples",
        "ASIANPAINT.NS": "Materials",
    }
    last["sector"] = last["ticker"].map(SECTOR_MAP).fillna("Other")
    sec_w = last.groupby("sector")["target_w"].sum()
    sector_cap_pct = 35.0
    pol = load_yaml(CONFIG) if CONFIG.exists() else {}
    sector_cap_pct = float(pol.get("sector_cap_base_pct", sector_cap_pct))
    sec_breach = sec_w > sector_cap_pct / 100.0
    checks.append(
        {
            "check": "sector_cap_breach_any",
            "value": bool(sec_breach.any()),
            "limit": f"<={sector_cap_pct / 100.0}",
        }
    )

    # ADV capacity check (if ADV available & portfolio value known)
    adv_ok = True
    if ADV_PQ.exists() and POSPQ.exists():
        adv = pd.read_parquet(ADV_PQ)
        adv["date"] = pd.to_datetime(adv["date"])
        # take last day ADV per ticker
        adv_day = adv[adv["date"] == pd.to_datetime(last_day)]
        if not adv_day.empty:
            # portfolio value from latest snapshot
            pos = pd.read_parquet(POSPQ)
            pos["date"] = pd.to_datetime(pos["date"])
            pv = float(
                pos.loc[pos["date"] == pos["date"].max(), "port_value"]
                .drop_duplicates()
                .iloc[-1]
            )
            last = last.merge(adv_day[["ticker", "adv_value"]], on="ticker", how="left")
            cap_pct = (
                float(pol.get("adv_cap_pct_L1", pol.get("adv_cap_pct", 10))) / 100.0
            )
            last["target_value"] = last["target_w"] * pv
            last["per_name_cap_value"] = cap_pct * last["adv_value"]
            last["adv_cap_breach"] = last["target_value"] > last["per_name_cap_value"]
            adv_ok = not bool(last["adv_cap_breach"].fillna(False).any())
            checks.append(
                {
                    "check": "adv_cap_breach_any",
                    "value": not adv_ok,
                    "limit": f"<= {cap_pct} * ADV",
                }
            )
    # Emit validation CSV
    vrep = pd.DataFrame(checks)
    vrep.to_csv(OUT_VCHECK, index=False)

    print(
        json.dumps(
            {
                "orders_schedule_csv": str(OUT_SCHEDULE),
                "orders_lastday_csv": str(OUT_TODAY),
                "validation_csv": str(OUT_VCHECK),
                "last_day": last_day,
                "per_name_cap_breach_any": bool(
                    vrep.loc[vrep["check"] == "per_name_cap_breach_any", "value"].iloc[
                        0
                    ]
                ),
                "sector_cap_breach_any": (
                    bool(
                        vrep.loc[
                            vrep["check"] == "sector_cap_breach_any", "value"
                        ].iloc[0]
                    )
                    if (vrep["check"] == "sector_cap_breach_any").any()
                    else False
                ),
            },
            indent=2,
        )
    )

    open_win(OUT_TODAY)
    open_win(OUT_SCHEDULE)
    open_win(OUT_VCHECK)


if __name__ == "__main__":
    main()

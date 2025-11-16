from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(r"F:\Projects\trading_stack_py")
REPORTS = ROOT / "reports"
CFG = ROOT / "config" / "event_rules.yaml"
CAL = REPORTS / "events_calendar.csv"
OUT = REPORTS / "events_position_flags.csv"


def open_win(p: Path):
    if sys.platform.startswith("win") and p.exists():
        os.startfile(p)


def load_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main():
    pos_pq = REPORTS / "positions_daily.parquet"
    if not pos_pq.exists():
        raise SystemExit("Missing reports\\positions_daily.parquet — run W3 bootstrap first.")
    if not CAL.exists():
        raise SystemExit("Missing events_calendar.csv — run w8_make_events_calendar.py first.")
    if not CFG.exists():
        raise SystemExit("Missing config\\event_rules.yaml")

    pol = load_yaml(CFG).get("policy", {})
    pre_d = int(pol.get("pre_event_freeze_days", 1))
    post_d = int(pol.get("post_event_hold_days", 1))
    mult_e = float(pol.get("earnings_reduce_multiplier", 0.75))
    block_adds = bool(pol.get("block_new_positions_pre_event", True))
    block_rebal = bool(pol.get("block_rebalance_on_event_day", True))

    hol_block = bool(pol.get("holiday_block_trading", True))
    hol_mult = float(pol.get("holiday_multiplier", 0.0))

    div_mult = float(pol.get("dividend_reduce_multiplier", 0.90))
    div_pre = int(pol.get("dividend_pre_days", 0))
    div_post = int(pol.get("dividend_post_days", 0))

    # positions dates universe
    pos = pd.read_parquet(pos_pq)[["date", "ticker", "weight", "port_value"]].copy()
    pos["date"] = pd.to_datetime(pos["date"])
    all_dates = pd.to_datetime(pos["date"].unique())
    tickers = pos["ticker"].unique()

    # calendar
    cal = pd.read_csv(CAL)
    cal["date"] = pd.to_datetime(cal["date"])

    # build per-day/ticker flags
    rows = []
    df_template = pd.DataFrame([(d, t) for d in sorted(all_dates) for t in tickers], columns=["date", "ticker"])
    df = df_template.merge(
        pos[["date", "ticker", "port_value"]].drop_duplicates(),
        on=["date", "ticker"],
        how="left",
    )

    # start with neutral
    df["allow_new"] = True
    df["rebalance_allowed"] = True
    df["risk_mult"] = 1.0
    df["event_note"] = ""

    # map of earnings per (ticker, date)
    earn = cal[cal["event_type"] == "earnings"][["date", "ticker", "session"]].copy()
    earn["event"] = "earnings"
    # holidays (no ticker)
    hol = cal[cal["event_type"] == "holiday"][["date"]].copy().drop_duplicates()
    hol["event"] = "holiday"

    # apply holiday rules
    if not hol.empty:
        idx = df["date"].isin(hol["date"])
        if hol_block:
            df.loc[idx, ["allow_new", "rebalance_allowed"]] = [False, False]
            df.loc[idx, "risk_mult"] = hol_mult
            df.loc[idx, "event_note"] = df.loc[idx, "event_note"] + "holiday;"
        else:
            df.loc[idx, "event_note"] = df.loc[idx, "event_note"] + "holiday_no_block;"

    # helper to apply windowed rule
    def apply_window(events_df, pre_days, post_days, reduce_mult, name):
        for _, r in events_df.iterrows():
            edate = r["date"]
            eticker = r.get("ticker", None)
            if pd.isna(eticker) or eticker == "":
                continue
            win = (
                (df["ticker"].eq(eticker))
                & (df["date"] >= edate - pd.tseries.offsets.BDay(pre_days))
                & (df["date"] <= edate + pd.tseries.offsets.BDay(post_days))
            )
            df.loc[win, "risk_mult"] = df.loc[win, "risk_mult"] * reduce_mult
            df.loc[win, "event_note"] = df.loc[win, "event_note"] + f"{name};"
            # pre-event freeze on adds
            if block_adds and pre_days > 0:
                pre_mask = (
                    (df["ticker"].eq(eticker))
                    & (df["date"] >= edate - pd.tseries.offsets.BDay(pre_days))
                    & (df["date"] < edate)
                )
                df.loc[pre_mask, "allow_new"] = False
            # event day rebalance block
            if block_rebal:
                ev_day = (df["ticker"].eq(eticker)) & (df["date"] == edate)
                df.loc[ev_day, "rebalance_allowed"] = False

    # earnings window
    apply_window(earn, pre_d, post_d, mult_e, "earnings")

    # dividends window (optional)
    div = cal[cal["event_type"] == "dividend"][["date", "ticker"]].copy()
    if not div.empty and (div_pre > 0 or div_post > 0):
        div["session"] = "full-day"
        apply_window(div, div_pre, div_post, div_mult, "dividend")

    # tidy + save
    df = df.sort_values(["date", "ticker"])
    REPORTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    print(
        json.dumps(
            {
                "out_csv": str(OUT),
                "rows": int(df.shape[0]),
                "unique_events": cal["event_type"].value_counts().to_dict(),
            },
            indent=2,
        )
    )

    open_win(OUT)
    open_win(CFG)
    open_win(REPORTS)


if __name__ == "__main__":
    main()

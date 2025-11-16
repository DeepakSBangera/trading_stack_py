# 2) FULL FILE CONTENTS — paste all below
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(r"F:\Projects\trading_stack_py")
SRC = ROOT / "reports" / "wk12_orders_lastday.csv"
OUT = ROOT / "reports" / "wk12_orders_lastday.norm.csv"

ALIASES = {
    "ticker": ["ticker", "symbol", "code", "instrument", "secid", "security", "name"],
    "side": ["side", "action", "direction", "trade_side", "order_side", "signal"],
    "qty": ["qty", "quantity", "shares", "units", "size", "qtty", "order_qty"],
    "px_ref": [
        "px_ref",
        "ref_price",
        "price",
        "px",
        "target_px",
        "reference_price",
        "close",
        "last",
        "open",
    ],
    "date": ["date", "day", "trade_date"],
}

SIDE_MAP = {
    "B": "BUY",
    "BUY": "BUY",
    "LONG": "BUY",
    "+": "BUY",
    "S": "SELL",
    "SELL": "SELL",
    "SHORT": "SELL",
    "-": "SELL",
}


def _find_col(cols, candidates):
    lc = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in lc:
            return lc[cand]
    # try fuzzy contains
    for c in lc:
        for cand in candidates:
            if cand in c:
                return lc[c]
    return None


def main():
    if not SRC.exists():
        print(f"Missing {SRC}")
        sys.exit(1)
    df = pd.read_csv(SRC)

    cols = list(df.columns)
    m = {}
    for key, cands in ALIASES.items():
        hit = _find_col([c.lower() for c in cols], [x.lower() for x in cands])
        if hit is not None:
            # map back to original case
            for c in cols:
                if c.lower() == hit:
                    m[key] = c
                    break

    missing_keys = [k for k in ["ticker", "side", "qty", "px_ref"] if k not in m]
    if missing_keys:
        print("Could not auto-map required columns:", missing_keys)
        print("Available columns:", cols)
        sys.exit(2)

    out = pd.DataFrame()
    out["ticker"] = df[m["ticker"]].astype(str).str.strip().str.upper()

    side_raw = df[m["side"]].astype(str).str.strip().str.upper()
    out["side"] = side_raw.map(SIDE_MAP).fillna(side_raw)
    # keep only BUY/SELL
    out = out[out["side"].isin(["BUY", "SELL"])].copy()

    # qty: force integer
    out["qty"] = pd.to_numeric(df[m["qty"]], errors="coerce").fillna(0).astype(int)

    # px_ref: numeric
    out["px_ref"] = pd.to_numeric(df[m["px_ref"]], errors="coerce")

    # date if present, else blank
    if "date" in m:
        out["date"] = pd.to_datetime(df[m["date"]], errors="coerce").dt.date
    else:
        out["date"] = pd.NaT

    # basic cleaning
    out = out[(out["qty"] != 0) & out["px_ref"].notna()].copy()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(
        {
            "in": str(SRC),
            "out": str(OUT),
            "rows": int(out.shape[0]),
            "cols": list(out.columns),
            "mapped": m,
        }
    )


if __name__ == "__main__":
    main()

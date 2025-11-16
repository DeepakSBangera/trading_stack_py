import json
from pathlib import Path

import pandas as pd

ROOT = Path(r"F:\Projects\trading_stack_py")
R = ROOT / "reports"
D = ROOT / "data" / "prices"
T = R / "wk11_blend_targets.csv"
V = R / "wk4_voltarget_stops.csv"


def pick(cols, cands):
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


df = pd.read_csv(T)
dcol = pick(df.columns, ["date", "dt", "trading_day", "asof", "as_of"])
tcol = pick(df.columns, ["ticker", "symbol", "name"])
df[dcol] = pd.to_datetime(df[dcol], errors="coerce").dt.date
last = df[dcol].dropna().max()
subset = set(df.loc[df[dcol] == last, tcol].astype(str))
have = set()
if (R / "wk4_voltarget_stops.csv").exists():
    v = pd.read_csv(V)
    have = set(v["ticker"].astype(str).unique())
missing = sorted(subset - have)
print(
    json.dumps(
        {
            "as_of": str(last),
            "tickers_input": len(subset),
            "with_atr": len(have),
            "missing": missing,
        },
        indent=2,
    )
)

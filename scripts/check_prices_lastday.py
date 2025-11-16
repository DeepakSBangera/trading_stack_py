import json
from pathlib import Path

import pandas as pd

ROOT = Path(r"F:\Projects\trading_stack_py")
REPORTS = ROOT / "reports"
DATA = ROOT / "data" / "prices"
df = pd.read_csv(REPORTS / "wk11_blend_targets.csv")


# pick flexible columns
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


dcol = pick(df.columns, ["date", "dt", "trading_day", "asof", "as_of"])
tcol = pick(df.columns, ["ticker", "symbol", "name", "secid", "instrument"])
df[dcol] = pd.to_datetime(df[dcol], errors="coerce").dt.date
last = df[dcol].dropna().max()
subs = df[df[dcol] == last][tcol].astype(str).unique().tolist()
miss = []
ok = []
for t in subs:
    p = DATA / f"{t}.parquet"
    if not p.exists():
        miss.append({"ticker": t, "why": "no_parquet"})
        continue
    try:
        x = pd.read_parquet(p)
        cols = {c.lower(): c for c in x.columns}
        dc = cols.get("date") or cols.get("dt")
        cc = cols.get("close") or cols.get("px_close") or cols.get("price")
        if not dc or not cc:
            miss.append({"ticker": t, "why": "no_date_or_close"})
            continue
        x[dc] = pd.to_datetime(x[dc], errors="coerce")
        v = x.loc[x[dc] == pd.to_datetime(last)]
        if v.empty and x[x[dc] < pd.to_datetime(last)].empty:
            miss.append({"ticker": t, "why": "no_prior_price"})
            continue
        ok.append(t)
    except Exception:
        miss.append({"ticker": t, "why": "read_error"})
print(
    json.dumps(
        {
            "last_day": str(last),
            "ok_count": len(ok),
            "miss_count": len(miss),
            "sample_missing": miss[:8],
        },
        indent=2,
    )
)

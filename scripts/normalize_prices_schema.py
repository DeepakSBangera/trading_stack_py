import json
from pathlib import Path

import pandas as pd

ROOT = Path(r"F:\Projects\trading_stack_py")
D = ROOT / "data" / "prices"
OUT = ROOT / "reports" / "normalize_prices_diag.json"


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


fixed = 0
scanned = 0
notes = []
for p in sorted(D.glob("*.parquet")):
    scanned += 1
    try:
        df = pd.read_parquet(p)
        cols = {c.lower(): c for c in df.columns}
        dcol = pick(df.columns, ["date", "dt", "timestamp"])
        ccol = pick(df.columns, ["close", "px_close", "price", "adj_close"])
        if dcol is None or ccol is None:
            notes.append({"file": p.name, "issue": "missing_date_or_close"})
            continue
        # Build normalized frame
        out = pd.DataFrame()
        out["date"] = pd.to_datetime(df[dcol], errors="coerce")
        out["close"] = pd.to_numeric(df[ccol], errors="coerce")
        # Optional OHLC best-effort
        for src, name in [("open", "open"), ("high", "high"), ("low", "low")]:
            s = pick(df.columns, [src, f"px_{src}", f"{src}_price"])
            if s:
                out[name] = pd.to_numeric(df[s], errors="coerce")
        if "open" not in out:
            out["open"] = out["close"]
        if "high" not in out:
            out["high"] = out[["open", "close"]].max(axis=1)
        if "low" not in out:
            out["low"] = out[["open", "close"]].min(axis=1)
        out = out.dropna(subset=["date", "close"]).sort_values("date")
        out.to_parquet(p, index=False)
        fixed += 1
    except Exception as e:
        notes.append({"file": p.name, "issue": "read_error", "err": str(e)})

OUT.write_text(
    json.dumps({"scanned": scanned, "fixed": fixed, "notes": notes}, indent=2),
    encoding="utf-8",
)
print(json.dumps({"scanned": scanned, "fixed": fixed}, indent=2))

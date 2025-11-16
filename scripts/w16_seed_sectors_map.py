from pathlib import Path

import pandas as pd

ROOT = Path(r"F:\Projects\trading_stack_py")
REPORTS = ROOT / "reports"
TARGETS = REPORTS / "wk11_blend_targets.csv"
SECTORS = REPORTS / "sectors_map.csv"


def main():
    if not TARGETS.exists():
        print(f"Missing {TARGETS}")
        return
    df = pd.read_csv(TARGETS)
    t = None
    for c in ["ticker", "symbol", "name"]:
        if c in df.columns:
            t = c
            break
    d = None
    for c in ["date", "dt", "trading_day"]:
        if c in df.columns:
            d = c
            break
    if d is not None:
        df[d] = pd.to_datetime(df[d], errors="coerce")
        last = df[d].max()
        df = df[df[d] == last]
    u = sorted(set(df[t].astype(str)))
    out = pd.DataFrame({"ticker": u, "sector": ["UNKNOWN"] * len(u)})
    REPORTS.mkdir(parents=True, exist_ok=True)
    out.to_csv(SECTORS, index=False)
    print({"sectors_csv": str(SECTORS), "tickers": len(u)})


if __name__ == "__main__":
    main()

# scripts/w6_enforce_caps.py
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(r"F:\Projects\trading_stack_py")
REPORTS = ROOT / "reports"
META = ROOT / "data" / "meta"

WE_CAP = REPORTS / "wk6_weights_capped.csv"
SECTORMAP = META / "sector_map.csv"
OUT_VAL = REPORTS / "wk6_caps_validation.csv"

SECTOR_CAP = 0.45
NAME_CAP = 0.10


def _load_sector_map():
    if SECTORMAP.exists():
        m = pd.read_csv(SECTORMAP)
        low = {c.lower(): c for c in m.columns}
        t = low.get("ticker")
        s = low.get("sector") or low.get("industry")
        if t and s:
            return {str(r[t]): str(r[s]) for _, r in m.iterrows()}
    return {}


def main():
    if not WE_CAP.exists():
        raise SystemExit("Missing wk6_weights_capped.csv. Run w6_optimizer_compare.py first.")
    df = pd.read_csv(WE_CAP)
    low = {c.lower(): c for c in df.columns}
    d = low.get("date")
    t = low.get("ticker")
    w = low.get("w_capped")
    sec = low.get("sector")
    if not d or not t or not w:
        raise SystemExit("wk6_weights_capped.csv missing columns.")
    if not sec:
        sm = _load_sector_map()
        df["sector"] = [sm.get(x, "UNKNOWN") for x in df[t].astype(str)]
        sec = "sector"

    # checks
    per_name = df.groupby([d, t], as_index=False)[w].apply(lambda s: s.abs().sum())
    per_name.columns = [d, t, "gross_w"]
    per_name["breach_name_cap"] = per_name["gross_w"] > NAME_CAP

    per_sector = df.groupby([d, sec], as_index=False)[w].apply(lambda s: s.abs().sum())
    per_sector.columns = [d, sec, "gross_w"]
    per_sector["breach_sector_cap"] = per_sector["gross_w"] > SECTOR_CAP

    # write combined view (for latest date)
    last = per_name[d].max()
    pn = per_name[per_name[d] == last].copy()
    ps = per_sector[per_sector[d] == last].copy()
    pn.to_csv(OUT_VAL, index=False)
    # append sector rows under a separator
    with open(OUT_VAL, "a", encoding="utf-8") as f:
        f.write("\n# sector_gross\n")
    ps.to_csv(OUT_VAL, mode="a", index=False)

    print(
        json.dumps(
            {
                "validation_csv": str(OUT_VAL),
                "name_cap_breaches": int(pn["breach_name_cap"].sum()),
                "sector_cap_breaches": int(ps["breach_sector_cap"].sum()),
                "as_of": str(last),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

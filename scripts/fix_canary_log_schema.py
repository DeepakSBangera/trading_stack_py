from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(r"F:\Projects\trading_stack_py")
REPORTS = ROOT / "reports"
SRC = REPORTS / "canary_log.csv"
BAK = REPORTS / "canary_log.bak.csv"

# We want at least these columns (case-insensitive). We'll coerce reasonable variants.
TARGET = ["date", "policy", "notional", "status"]

ALIASES = {
    "date": {
        "date",
        "dt",
        "run_date",
        "as_of",
        "as_of_ist",
        "as_of_utc",
        "ts",
        "ts_utc",
        "timestamp",
    },
    "policy": {"policy", "mode", "profile", "portfolio"},
    "notional": {
        "notional",
        "notional_inr",
        "size",
        "capital",
        "capital_inr",
        "nav_slice",
    },
    "status": {"status", "state", "stage", "phase", "result"},
}


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    cols_lc = {c: c.lower().strip() for c in df.columns}
    inv = {}
    for k, v in cols_lc.items():
        inv[v] = k  # last wins

    out = pd.DataFrame(index=df.index)
    for tgt in TARGET:
        found = None
        for c_lc in ALIASES[tgt]:
            if c_lc in inv:
                found = inv[c_lc]
                break
        if found is not None:
            out[tgt] = df[found]
        else:
            # fill missing with defaults
            if tgt == "date":
                out[tgt] = pd.NaT
            elif tgt == "notional":
                out[tgt] = 0.0
            else:
                out[tgt] = ""
    # type coercions
    if out["date"].isna().all():
        # try to parse any datetime-like columns from original
        for c in df.columns:
            try:
                parsed = pd.to_datetime(df[c], errors="coerce")
                if parsed.notna().any():
                    out["date"] = parsed.dt.tz_localize(None)
                    break
            except Exception:
                pass
    else:
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.tz_localize(None)

    # numeric notional
    out["notional"] = pd.to_numeric(out["notional"], errors="coerce").fillna(0.0)

    # final ordering
    return out[TARGET]


def main() -> None:
    if not SRC.exists():
        print({"ok": False, "notes": f"{SRC} missing"})
        return
    df = pd.read_csv(SRC)
    norm = _normalize(df)
    # backup original then overwrite
    if SRC.exists():
        SRC.replace(BAK)
    norm.to_csv(SRC, index=False)
    print(
        {
            "ok": True,
            "rows": int(norm.shape[0]),
            "columns": TARGET,
            "backup": str(BAK),
            "normalized": str(SRC),
        }
    )


if __name__ == "__main__":
    main()

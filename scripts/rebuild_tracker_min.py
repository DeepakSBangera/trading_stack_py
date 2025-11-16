from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DOCS = ROOT / "docs"
TRACKER = DOCS / "living_tracker.csv"
BACKUP = ROOT / "reports" / "living_tracker.rebuild.bak.csv"

REQ_COLS = ["session", "ts_ist", "note"]
WK_SUMMARY_RE = re.compile(r"^wk(\d{1,2})_.*_summary\.json$", re.IGNORECASE)


def _rows_from_reports() -> list[dict]:
    rows: list[dict] = []
    if not REPORTS.exists():
        return rows

    for p in REPORTS.glob("wk*_*.json"):
        m = WK_SUMMARY_RE.match(p.name)
        if not m:
            continue
        week = int(m.group(1))
        session = f"S-W{week}"

        ts = None
        note = p.stem.replace("_summary", "")
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            for k in ("as_of_ist", "as_of", "timestamp_ist", "timestamp"):
                v = data.get(k)
                if isinstance(v, str) and v.strip():
                    t = pd.to_datetime(v, errors="coerce")
                    if not pd.isna(t):
                        ts = t
                        break
            if isinstance(data.get("notes"), str) and data["notes"].strip():
                note = data["notes"].strip()
        except Exception:
            pass

        if ts is None or pd.isna(ts):
            # fallback: file mtime (UTC)
            ts = pd.to_datetime(p.stat().st_mtime, unit="s", utc=True)

        # Normalize to IST no matter what we got
        if getattr(ts, "tzinfo", None) is None:
            ts = ts.tz_localize("Asia/Kolkata")
        else:
            ts = ts.tz_convert("Asia/Kolkata")

        rows.append({"session": session, "ts_ist": ts.isoformat(), "note": note})

    if not rows:
        return []
    df = pd.DataFrame(rows).sort_values("ts_ist").groupby("session", as_index=False).tail(1).sort_values("ts_ist")
    return df.to_dict(orient="records")


def _to_ist_series(s: pd.Series) -> pd.Series:
    """Robust convert mixed tz/naive strings to IST-aware timestamps."""

    def conv(x):
        if pd.isna(x):
            return pd.NaT
        t = pd.to_datetime(x, errors="coerce")
        if pd.isna(t):
            return pd.NaT
        if getattr(t, "tzinfo", None) is None:
            t = t.tz_localize("Asia/Kolkata")
        else:
            t = t.tz_convert("Asia/Kolkata")
        return t

    out = s.apply(conv)
    # Fill NaT with now() just to avoid empty values; they’ll sort to end.
    now_ist = pd.Timestamp.now(tz="Asia/Kolkata")
    return out.fillna(now_ist)


def main():
    DOCS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    if TRACKER.exists():
        BACKUP.write_bytes(TRACKER.read_bytes())

    base = pd.DataFrame(columns=REQ_COLS)
    if TRACKER.exists():
        try:
            tmp = pd.read_csv(TRACKER, dtype=str, engine="python", on_bad_lines="skip")
            tmp.columns = [c.strip() for c in tmp.columns]
            if set(REQ_COLS).issubset(tmp.columns):
                base = tmp[REQ_COLS].copy()
        except Exception:
            pass

    new_rows = _rows_from_reports()
    new = pd.DataFrame(new_rows, columns=REQ_COLS)

    out = pd.concat([base, new], ignore_index=True)
    if out.empty:
        out = pd.DataFrame(columns=REQ_COLS)

    out["session"] = out.get("session", "").astype(str).str.strip()
    out["note"] = (
        out.get("note", "")
        .astype(str)
        .str.replace("\r", " ", regex=False)
        .str.replace("\n", " ", regex=False)
        .str.strip()
    )

    # Convert to tz-aware IST, then serialize using per-element .isoformat()
    ts_series = _to_ist_series(out.get("ts_ist", pd.Series(dtype="object")))
    out["ts_ist"] = ts_series.apply(lambda t: t.isoformat())

    out = out.drop_duplicates(subset=REQ_COLS, keep="last").sort_values("ts_ist").reset_index(drop=True)

    out.to_csv(TRACKER, index=False)
    print(
        {
            "ok": True,
            "tracker": str(TRACKER),
            "backup": str(BACKUP if BACKUP.exists() else ""),
            "rows": int(out.shape[0]),
        }
    )


if __name__ == "__main__":
    main()

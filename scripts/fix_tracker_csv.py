from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRACKER = ROOT / "docs" / "living_tracker.csv"
BACKUP = ROOT / "reports" / "living_tracker.bak.csv"

# Minimal schema we keep in the tracker
REQ_COLS = ["session", "ts_ist", "note"]


def _safe_read_tracker(path: Path) -> pd.DataFrame:
    """
    Read the tracker CSV permissively:
    - tolerate extra commas / malformed rows
    - coerce everything to str so we can clean later
    """
    if not path.exists():
        # start an empty frame with required columns
        return pd.DataFrame(columns=REQ_COLS)

    df = pd.read_csv(
        path,
        engine="python",  # more tolerant
        on_bad_lines="skip",  # drop broken rows
        quoting=csv.QUOTE_MINIMAL,
        dtype=str,
    )
    # Normalize column names to lowercase for mapping
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _map_required_cols(df: pd.DataFrame) -> pd.DataFrame:
    cols_lower = {c.lower(): c for c in df.columns}

    def col_or_none(name: str) -> str | None:
        return cols_lower.get(name, None)

    session_col = col_or_none("session")
    ts_col = col_or_none("ts_ist")
    note_col = col_or_none("note")

    out = pd.DataFrame(columns=REQ_COLS)
    out["session"] = df[session_col] if session_col else "S-UNKNOWN"
    out["ts_ist"] = df[ts_col] if ts_col else pd.NaT
    out["note"] = df[note_col] if note_col else ""

    return out


def _clean_types(out: pd.DataFrame) -> pd.DataFrame:
    # Timestamp: parse to timezone-aware IST; fill missing with "now"
    ts = pd.to_datetime(out["ts_ist"], errors="coerce", utc=False)
    # Make naive → IST; if already tz-aware, convert to IST
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize("Asia/Kolkata", nonexistent="NaT", ambiguous="NaT")
    else:
        ts = ts.dt.tz_convert("Asia/Kolkata")
    ts = ts.fillna(pd.Timestamp.now(tz="Asia/Kolkata"))
    out["ts_ist"] = ts

    # Session & note cleanup
    out["session"] = out["session"].astype(str).str.strip().replace({"": "S-UNKNOWN"})
    out["note"] = (
        out["note"]
        .astype(str)
        .str.replace("\r", " ", regex=False)
        .str.replace("\n", " ", regex=False)
        .str.replace("\t", " ", regex=False)
        .str.strip()
    )

    # Deduplicate exact rows
    out = out.drop_duplicates(subset=REQ_COLS, keep="last")

    # Ensure column order
    return out[REQ_COLS]


def main():
    TRACKER.parent.mkdir(parents=True, exist_ok=True)
    BACKUP.parent.mkdir(parents=True, exist_ok=True)

    # Backup (even if empty file, this is fine)
    if TRACKER.exists():
        BACKUP.write_bytes(TRACKER.read_bytes())

    df_raw = _safe_read_tracker(TRACKER)
    cleaned = _map_required_cols(df_raw)
    cleaned = _clean_types(cleaned)

    cleaned.to_csv(TRACKER, index=False)
    print(
        {
            "ok": True,
            "backup": str(BACKUP if BACKUP.exists() else ""),
            "tracker": str(TRACKER),
            "rows_in": int(df_raw.shape[0]),
            "rows_out": int(cleaned.shape[0]),
            "cols": list(cleaned.columns),
        }
    )


if __name__ == "__main__":
    main()

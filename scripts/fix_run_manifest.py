from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

ROOT = Path(r"F:\Projects\trading_stack_py")
REPORTS = ROOT / "reports"
CONFIG = ROOT / "config"
MANIFEST = REPORTS / "run_manifest.jsonl"
BACKUP = REPORTS / "run_manifest.bak.jsonl"
CLEANED = REPORTS / "run_manifest.cleaned.jsonl"

REQ = ["ts_utc", "git_sha", "artifact", "config_hash"]


def _git_sha() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short=8", "HEAD"], cwd=ROOT, text=True).strip()
        return out or "????????"
    except Exception:
        return "????????"


def _first_present(d: Dict[str, Any], keys: List[str]) -> Any:
    for k in keys:
        if k in d and d[k] not in (None, "", {}):
            return d[k]
    return None


def _hash_configs() -> str:
    # lightweight: hash filenames+sizes+mtimes so changes flip the hash without needing PyYAML
    h = hashlib.sha256()
    if CONFIG.exists():
        for p in sorted(CONFIG.rglob("*")):
            if p.is_file():
                st = p.stat()
                h.update(str(p.relative_to(CONFIG)).encode())
                h.update(str(st.st_size).encode())
                h.update(str(int(st.st_mtime)).encode())
    return h.hexdigest()[:16]


def _utc_now_iso() -> str:
    # Always produce a tz-aware UTC ISO string
    return pd.Timestamp.now(tz="UTC").isoformat()


def _maybe_ts(row: Dict[str, Any]) -> str:
    """
    Try to derive a UTC ISO timestamp from common fields or artifact mtime; else now(UTC).
    """
    cand = _first_present(row, ["ts_utc", "ts", "timestamp", "as_of_utc", "as_of"])
    if cand:
        try:
            # utc=True parses as UTC if naive, or preserves tz if present, then convert to UTC
            return pd.to_datetime(cand, utc=True).tz_convert("UTC").isoformat()
        except Exception:
            pass

    # infer from artifact mtime if possible
    art = str(row.get("artifact", "")).strip()
    if art:
        p = Path(art)
        if not p.is_absolute():
            p = (ROOT / art).resolve()
        if p.exists():
            try:
                return pd.Timestamp(p.stat().st_mtime, unit="s", tz="UTC").isoformat()
            except Exception:
                pass

    # fallback: now UTC
    return _utc_now_iso()


def _maybe_artifact(row: Dict[str, Any]) -> str:
    # pull from common keys or embedded 'files'
    cand = _first_present(row, ["artifact", "report", "path"])
    if cand:
        return str(cand)
    files = row.get("files") or {}
    for k in ["detail_csv", "summary_json", "zip", "wire_csv", "html"]:
        if k in files:
            return str(files[k])
    return "unknown"


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    if not MANIFEST.exists():
        raise SystemExit(f"Missing {MANIFEST}")

    rows: List[Dict[str, Any]] = []
    with open(MANIFEST, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rows.append(json.loads(ln))
            except Exception:
                # skip corrupt lines to avoid breaking conversion
                pass

    sha = _git_sha()
    cfg_h = _hash_configs()

    cleaned: List[Dict[str, Any]] = []
    for r in rows:
        out = dict(r)  # copy original
        out.setdefault("git_sha", sha)
        out.setdefault("artifact", _maybe_artifact(out))
        out.setdefault("ts_utc", _maybe_ts(out))
        out.setdefault("config_hash", cfg_h)

        # Normalize ts_utc robustly to tz-aware UTC ISO
        try:
            out["ts_utc"] = pd.to_datetime(out["ts_utc"], utc=True).tz_convert("UTC").isoformat()
        except Exception:
            out["ts_utc"] = _utc_now_iso()

        cleaned.append(out)

    # backup original, write cleaned, then swap in place
    if MANIFEST.exists():
        MANIFEST.replace(BACKUP)
    with open(CLEANED, "w", encoding="utf-8") as f:
        for r in cleaned:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    CLEANED.replace(MANIFEST)

    print(
        {
            "rows_in": len(rows),
            "rows_out": len(cleaned),
            "git_sha": sha,
            "config_hash": cfg_h,
            "backup": str(BACKUP),
            "manifest": str(MANIFEST),
        }
    )


if __name__ == "__main__":
    main()

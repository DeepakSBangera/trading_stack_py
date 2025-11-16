# scripts/w10_lineage_manifest_check.py
from __future__ import annotations

import csv
import datetime as dt
import json
from pathlib import Path
from typing import Any, List

ROOT = Path(r"F:\Projects\trading_stack_py")
REPORTS = ROOT / "reports"

MAN_JSONL = REPORTS / "run_manifest.jsonl"
MAN_LAST = REPORTS / "run_manifest_last.json"
MAN_INDEX = REPORTS / "run_manifest_index.csv"
OUT_TABLE = REPORTS / "w10_lineage_report.csv"
OUT_SUM = REPORTS / "w10_lineage_diag.json"

# ---- helpers -----------------------------------------------------------------


def _iter_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
                if isinstance(obj, dict):
                    rows.append(obj)
            except Exception:
                pass
    return rows


def _stat_file(p: Path) -> dict:
    d = {"path": str(p), "exists": p.exists(), "size_bytes": None, "mtime": None}
    if p.exists():
        st = p.stat()
        d["size_bytes"] = int(st.st_size)
        d["mtime"] = dt.datetime.fromtimestamp(st.st_mtime).isoformat(
            timespec="seconds"
        )
    return d


_POSSIBLE_KEYS = [
    "path",
    "file",
    "artifact",
    "artifact_path",
    "output",
    "output_path",
    "relpath",
    "abspath",
]


def _extract_path_from_obj(x: Any) -> str | None:
    """
    Return a string path from:
      - str
      - dict with one of _POSSIBLE_KEYS
      - dict with single key whose value looks like a path
    """
    if isinstance(x, str):
        return x.strip() or None
    if isinstance(x, dict):
        # direct key match
        for k in _POSSIBLE_KEYS:
            if k in x and isinstance(x[k], str) and x[k].strip():
                return x[k].strip()
        # try common nested shapes: {"name": "...", "path": {...}} etc. (flatten best-effort)
        for v in x.values():
            if isinstance(v, str) and v.strip():
                # prefer strings that look like files
                return v.strip()
        # last resort: ignore non-stringy dicts
        return None
    # lists inside artifacts (rare) -> take first usable
    if isinstance(x, (list, tuple)):
        for e in x:
            s = _extract_path_from_obj(e)
            if s:
                return s
        return None
    return None


def _normalize_artifacts(a: Any) -> List[str]:
    """
    Accepts list/dict/str and returns list[str] of paths.
    """
    out: List[str] = []
    if a is None:
        return out
    if isinstance(a, dict):
        # values might be strings or dicts
        for v in a.values():
            s = _extract_path_from_obj(v)
            if s:
                out.append(s)
        return out
    if isinstance(a, (list, tuple)):
        for e in a:
            s = _extract_path_from_obj(e)
            if s:
                out.append(s)
        return out
    if isinstance(a, str):
        s = a.strip()
        if s:
            out.append(s)
    return out


# ---- main --------------------------------------------------------------------


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)

    entries = _iter_jsonl(MAN_JSONL)
    if entries:
        latest = entries[-1]
    elif MAN_LAST.exists():
        try:
            latest = json.loads(MAN_LAST.read_text(encoding="utf-8"))
        except Exception:
            latest = {}
    else:
        latest = {}

    # artifacts might be under various keys depending on producer
    raw_artifacts = (
        latest.get("artifacts") or latest.get("outputs") or latest.get("files") or []
    )
    artifacts = _normalize_artifacts(raw_artifacts)

    rows = []
    missing = 0
    for raw in artifacts:
        p = Path(raw)
        if not p.is_absolute():
            p = ROOT / raw
        st = _stat_file(p)
        rows.append(st)
        if not st["exists"]:
            missing += 1

    # write artifact table
    with OUT_TABLE.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "exists", "size_bytes", "mtime"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    diag = {
        "manifest_jsonl_exists": MAN_JSONL.exists(),
        "manifest_last_exists": MAN_LAST.exists(),
        "manifest_index_exists": MAN_INDEX.exists(),
        "artifacts_count": len(rows),
        "artifacts_missing": missing,
    }
    OUT_SUM.write_text(json.dumps(diag, indent=2), encoding="utf-8")
    print(json.dumps({"report_csv": str(OUT_TABLE), **diag}, indent=2))


if __name__ == "__main__":
    main()

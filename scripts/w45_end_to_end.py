from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(r"F:\Projects\trading_stack_py")
DOCS = ROOT / "docs"
REPORTS = ROOT / "reports"
CONFIG = ROOT / "config"
SCRIPTS = ROOT / "scripts"

RUN_MANIFEST = REPORTS / "run_manifest.jsonl"
TRACKER = DOCS / "living_tracker.csv"
SUMMARY_JSON = REPORTS / "wk45_e2e_summary.json"


def _utc_now_iso():
    return datetime.now(UTC).isoformat()


def _safe_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for _ in f)


def _tracker_rows(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        df = pd.read_csv(path)
        return int(df.shape[0])
    except Exception:
        return 0


def _latest_reports(n=10):
    REPORTS.mkdir(parents=True, exist_ok=True)
    files = sorted(REPORTS.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    out = []
    for p in files[:n]:
        try:
            out.append({"name": p.name, "bytes": p.stat().st_size})
        except FileNotFoundError:
            pass
    return out


def _sanity():
    want = [
        REPORTS,
        DOCS,
        CONFIG,
        SCRIPTS,
    ]
    miss = [str(p) for p in want if not p.exists()]
    return {"ok": len(miss) == 0, "missing": miss}


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)

    sanity = _sanity()
    manifest_rows = _safe_jsonl_rows(RUN_MANIFEST)
    tracker_rows = _tracker_rows(TRACKER)

    payload = {
        "as_of_utc": _utc_now_iso(),
        "e2e_checks": sanity,
        "counts": {
            "run_manifest_rows": manifest_rows,
            "tracker_rows": tracker_rows,
        },
        "latest_reports": _latest_reports(12),
        "notes": "W45 end-to-end dry run summary. Ensure red-team (W44) is green before production freeze.",
        "files": {
            "summary_json": str(SUMMARY_JSON),
            "run_manifest": str(RUN_MANIFEST),
            "tracker_csv": str(TRACKER),
        },
    }

    SUMMARY_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

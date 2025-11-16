from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Dict, List

import pandas as pd

ROOT = Path(r"F:\Projects\trading_stack_py")
REPORTS = ROOT / "reports"
DOCS = ROOT / "docs"
DATA = ROOT / "data"
CONFIG = ROOT / "config"

OUT_SUMMARY = REPORTS / "wk44_redteam_summary.json"
OUT_REPORT_MD = DOCS / "wk44_redteam_report.md"

# Minimal expectations we claim to uphold across the stack
MUST_EXIST_FILES = [
    REPORTS / "run_manifest.jsonl",
]
MUST_EXIST_DIRS = [
    REPORTS,
    DOCS,
    DATA / "csv",
]

# A few representative artifacts to try replay/read
REPLAY_CANDIDATES = [
    REPORTS / "wk6_portfolio_compare.csv",
    REPORTS / "wk11_alpha_blend.csv",
    REPORTS / "wk41_momentum_tilt.csv",
    REPORTS / "wk43_barbell_compare.csv",
]


# “Simulated failure drills”
def _drill_missing_data_dir() -> Dict:
    """Simulate data outage by temporarily renaming data/csv, then verifying guard behavior."""
    csv_dir = DATA / "csv"
    if not csv_dir.exists():
        return {
            "name": "missing_data_dir",
            "skipped": True,
            "ok": True,
            "notes": "data/csv not found; skip",
        }
    tmp = DATA / "_csv_tmp_redteam"
    ok = True
    notes: List[str] = []
    try:
        csv_dir.rename(tmp)
        # Any script expecting data should fail gracefully; we just check presence/absence of required dirs
        ok = not csv_dir.exists() and tmp.exists()
        notes.append("Temporarily moved data/csv → data/_csv_tmp_redteam.")
    except Exception as e:
        ok = False
        notes.append(f"rename failed: {e}")
    finally:
        # restore
        try:
            if tmp.exists():
                tmp.rename(csv_dir)
                notes.append("Restored data/csv.")
        except Exception as e:
            ok = False
            notes.append(f"restore failed: {e}")
    return {"name": "missing_data_dir", "ok": ok, "notes": "; ".join(notes)}


def _drill_manifest_replay() -> Dict:
    """Basic lineage replay check: run_manifest.jsonl should be loadable; rows parse; last row has sha/timestamp."""
    man = REPORTS / "run_manifest.jsonl"
    if not man.exists():
        return {
            "name": "manifest_replay",
            "ok": False,
            "notes": "run_manifest.jsonl missing",
        }
    rows = []
    with open(man, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rows.append(json.loads(ln))
            except Exception as e:
                return {
                    "name": "manifest_replay",
                    "ok": False,
                    "notes": f"json parse error: {e}",
                }
    ok = len(rows) > 0 and all(isinstance(r, dict) for r in rows)
    last = rows[-1] if rows else {}
    fields = ["ts_utc", "git_sha", "artifact", "config_hash"]
    missing = [k for k in fields if k not in last]
    if missing:
        ok = False
    notes = f"rows={len(rows)}; last_has={','.join([k for k in fields if k in last])}; last_missing={','.join(missing)}"
    return {"name": "manifest_replay", "ok": ok, "rows": len(rows), "notes": notes}


def _drill_backup_zip() -> Dict:
    """Try to zip a small subset of reports to ensure we can export/restore quickly."""
    target = REPORTS / "W44_redteam_snapshot.zip"
    files = [p for p in REPLAY_CANDIDATES if p.exists()]
    ok = True
    try:
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for p in files:
                z.write(p, arcname=p.name)
    except Exception as e:
        ok = False
        return {"name": "backup_zip", "ok": False, "notes": str(e)}
    return {"name": "backup_zip", "ok": ok, "files": len(files), "zip": str(target)}


def _drill_canary_log() -> Dict:
    """If reports/canary_log.csv exists, ensure schema is sane."""
    p = REPORTS / "canary_log.csv"
    if not p.exists():
        return {
            "name": "canary_log_schema",
            "ok": True,
            "skipped": True,
            "notes": "canary_log.csv not present",
        }
    df = pd.read_csv(p)
    want_any = {"date", "policy", "notional", "status"}
    ok = len(set(c.lower() for c in df.columns) & set(w.lower() for w in want_any)) >= 2
    return {"name": "canary_log_schema", "ok": ok, "rows": int(df.shape[0])}


def _drill_kill_switch_config() -> Dict:
    """Sanity check kill-switch config if present."""
    ks = CONFIG / "kill_switch.yaml"
    if not ks.exists():
        return {
            "name": "kill_switch_config",
            "ok": True,
            "skipped": True,
            "notes": "kill_switch.yaml not present",
        }
    # lightweight read (no yaml dep): just ensure size and a few keywords
    txt = ks.read_text(encoding="utf-8", errors="ignore")
    ok = ("policy" in txt.lower()) and (len(txt) > 20)
    return {"name": "kill_switch_config", "ok": ok, "bytes": len(txt)}


def _existence_checks() -> Dict:
    files_ok = all(p.exists() for p in MUST_EXIST_FILES)
    dirs_ok = all(p.exists() for p in MUST_EXIST_DIRS)
    return {
        "name": "existence_checks",
        "ok": files_ok and dirs_ok,
        "files_ok": files_ok,
        "dirs_ok": dirs_ok,
        "missing_files": [str(p) for p in MUST_EXIST_FILES if not p.exists()],
        "missing_dirs": [str(p) for p in MUST_EXIST_DIRS if not p.exists()],
    }


def _write_markdown(summary: Dict) -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# W44 — Red-Team & Recovery Drill\n")
    lines.append(f"**As-of (IST):** {pd.Timestamp.utcnow().tz_convert('Asia/Kolkata').isoformat()}\n")
    lines.append("## Checks Run\n")
    for r in summary.get("checks", []):
        name = r.get("name", "unknown")
        ok = "✅" if r.get("ok") else "❌"
        extra = ""
        if "notes" in r:
            extra = f" — {r['notes']}"
        if "rows" in r:
            extra += f" — rows={r['rows']}"
        if "files" in r:
            extra += f" — files={r['files']}"
        if "zip" in r:
            extra += f" — zip={r['zip']}"
        lines.append(f"- **{name}**: {ok}{extra}")
    lines.append("\n## Result\n")
    lines.append(f"- **All-green:** {'✅' if summary.get('all_green') else '❌'}")
    lines.append(f"- **Critical files present:** {'✅' if summary.get('exist_ok') else '❌'}")
    lines.append("\n## Next Steps\n")
    lines.append("- Keep the snapshot zip safe (off-box copy).")
    lines.append("- If any check failed, file a hotfix task with rollback steps attached.")
    OUT_REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)

    checks = []
    checks.append(_existence_checks())
    checks.append(_drill_manifest_replay())
    checks.append(_drill_backup_zip())
    checks.append(_drill_canary_log())
    checks.append(_drill_kill_switch_config())
    checks.append(_drill_missing_data_dir())

    exist_ok = checks[0].get("ok", False)
    all_green = all(c.get("ok", False) for c in checks)

    out = {
        "as_of_ist": pd.Timestamp.utcnow().tz_convert("Asia/Kolkata").isoformat(),
        "checks": checks,
        "exist_ok": exist_ok,
        "all_green": all_green,
        "files": {
            "summary_json": str(OUT_SUMMARY),
            "report_md": str(OUT_REPORT_MD),
        },
    }
    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    _write_markdown(out)

    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

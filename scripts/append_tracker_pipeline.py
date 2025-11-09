from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from datetime import datetime, timezone

# --- repo layout ---
ROOT = Path(r"F:\Projects\trading_stack_py")
DOCS = ROOT / "docs"
REPORTS = ROOT / "reports"
INFO_DIR = REPORTS / "_run_info"  # per-script snapshots written by run_with_info.py-like wrappers

TRACKER = DOCS / "living_tracker.csv"  # canonical tracker

# default assumptions
DEFAULT_HOURS_PER_SESSION = 4.0  # your 4-hours/week tempo

def _iso_now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _ensure_tracker():
    DOCS.mkdir(parents=True, exist_ok=True)
    if not TRACKER.exists():
        # header is intentionally stable across weeks
        TRACKER.write_text(
            "date,session,hours,artifacts,gates,risks,decisions\n",
            encoding="utf-8",
        )

def _bool_flag(path: Path) -> str:
    return "present" if path.exists() else "missing"

def _read_freeze_status() -> dict:
    p = REPORTS / "w20_freeze_status.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}

def _count_artifacts() -> int:
    """
    Light artifact count: csv/parquet/json under reports (non-recursive except wires & _run_info).
    """
    if not REPORTS.exists():
        return 0
    n = 0
    for p in REPORTS.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".csv", ".parquet", ".json", ".yaml", ".yml", ".zip"}:
            n += 1
    return n

def _gates_status() -> str:
    gates = REPORTS / "wk0_gates.csv"
    return _bool_flag(gates)

def _risk_status() -> str:
    kill_yaml = REPORTS / "kill_switch.yaml"
    return _bool_flag(kill_yaml)

def _decision_line() -> str:
    """
    Human-readable one-liner from freeze or manifest status if available.
    """
    fz = _read_freeze_status()
    if fz.get("freeze_active") is True:
        return "freeze_active"
    if fz.get("freeze_on") is True and fz.get("in_window") is True:
        return "freeze_window"
    return ""

def write_script_info(name: str, session: str, purpose: str, inputs: list[str], outputs: list[str]) -> Path:
    """
    Optional helper: record a small per-script info JSON so lineage feels consistent
    even when not using the full run_manifest wrapper.
    """
    INFO_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": _iso_now_utc(),
        "session": session,
        "script": f"scripts/{name}",
        "purpose": purpose,
        "inputs": inputs,
        "outputs": outputs,
    }
    out_path = INFO_DIR / f"{name}.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path

def append_tracker_row(session: str, hours: float | None = None) -> dict:
    _ensure_tracker()
    dt_today = datetime.now().strftime("%Y-%m-%d")

    hours_done = float(hours) if hours is not None else DEFAULT_HOURS_PER_SESSION
    artifacts = _count_artifacts()
    gates = _gates_status()
    risks = _risk_status()
    decisions = _decision_line()

    row = {
        "date": dt_today,
        "session": session,
        "hours": f"{hours_done:.2f}",
        "artifacts": str(artifacts),
        "gates": gates,
        "risks": risks,
        "decisions": decisions,
    }

    # append CSV
    with TRACKER.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["date", "session", "hours", "artifacts", "gates", "risks", "decisions"])
        w.writerow(row)

    return {
        "tracker_csv": str(TRACKER),
        "session": session,
        "hours": hours_done,
        "artifacts_count": artifacts,
        "gates": gates,
        "risks": risks,
        "decisions": decisions,
    }

def _env_hours() -> float | None:
    val = os.environ.get("TRACKER_HOURS")
    if not val:
        return None
    try:
        return float(val)
    except Exception:
        return None

def main():
    # Session name from env or fallback
    session = os.environ.get("SESSION", "").strip() or "S-PIPELINE"
    hours = _env_hours()

    summary = append_tracker_row(session=session, hours=hours)

    # also write a tiny info stub so this script is discoverable
    write_script_info(
        name="append_tracker_pipeline.py",
        session=session,
        purpose="Append a single tracker line summarizing artifacts/gates/risks/decisions",
        inputs=["reports/* (scanned)"],
        outputs=[str(TRACKER)],
    )

    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()

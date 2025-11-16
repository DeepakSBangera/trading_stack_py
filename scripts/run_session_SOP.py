# scripts/run_session_SOP.py
from __future__ import annotations

import datetime as dt
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"F:\Projects\trading_stack_py")
PY = ROOT / ".venv" / "Scripts" / "python.exe"
WRAP = ROOT / "scripts" / "run_with_info.py"
REPORTS = ROOT / "reports"

# Ordered pipeline (script, session tag)
STEPS = [
    ("scripts\\w7_compute_regimes.py", "S-W7"),
    ("scripts\\w7_apply_gates.py", "S-W7"),
    ("scripts\\w8_apply_event_rules.py", "S-W8"),
    ("scripts\\w8_combine_schedules.py", "S-W8"),
    ("scripts\\w11_build_targets.py", "S-W11"),
    ("scripts\\w12_size_and_orders.py", "S-W12"),
]


def run_step(script_rel: str, session: str) -> int:
    """
    Run one step via wrapper with:
      - no auto-open to avoid file locks
      - manifest OFF per step (we add one pipeline-level manifest at end)
    """
    cmd = [
        str(PY),
        str(WRAP),
        script_rel,
        "--session",
        session,
        "--open",
        "none",
        "--manifest",
        "off",
    ]
    print(f"\n=== RUN: {script_rel} [{session}] ===")
    return subprocess.call(cmd, cwd=str(ROOT))


def main():
    start = dt.datetime.now()
    rc_any_fail = False

    # 1) Run pipeline steps
    for script_rel, sess in STEPS:
        rc = run_step(script_rel, sess)
        if rc != 0:
            rc_any_fail = True
            print(f"[warn] step failed (rc={rc}): {script_rel}")
            # continue so we still log cards/registry for earlier steps

    # 2) Write single pipeline-level manifest snapshot (atomic + retry inside w_manifest.py)
    manifest = ROOT / "scripts" / "w_manifest.py"
    if manifest.exists():
        subprocess.call([str(PY), str(manifest), "S-PIPELINE"], cwd=str(ROOT))

    # 3) Build a W12 review ZIP if helper exists
    w12_zip = ROOT / "scripts" / "w12_build_review_zip.py"
    if w12_zip.exists():
        subprocess.call([str(PY), str(w12_zip)], cwd=str(ROOT))

    # 4) Append one pipeline summary line to tracker if helper exists
    tracker_append = ROOT / "scripts" / "append_tracker_pipeline.py"
    if tracker_append.exists():
        subprocess.call([str(PY), str(tracker_append)], cwd=str(ROOT))

    # 5) Open living views (registry CSV + MD)
    reg = REPORTS / "script_registry.csv"
    living = ROOT / "docs" / "SCRIPTS_LIVING.md"
    try:
        subprocess.run(["cmd", "/c", "start", "", str(reg)], check=False)
        subprocess.run(["cmd", "/c", "start", "", str(living)], check=False)
    except Exception:
        pass

    dur = (dt.datetime.now() - start).total_seconds()
    print(f"\nDone. Duration: {dur:.1f}s  Errors: {int(rc_any_fail)}")
    sys.exit(1 if rc_any_fail else 0)


if __name__ == "__main__":
    main()

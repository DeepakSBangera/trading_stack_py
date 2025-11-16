# 2) FULL FILE CONTENTS — paste all below
# scripts/_include_manifest.py
from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(r"F:\Projects\trading_stack_py")


def record_manifest(session_label: str = "") -> None:
    """
    Lightweight helper to append a manifest snapshot from any script:
        from _include_manifest import record_manifest
        record_manifest("S-W12")
    """
    py = ROOT / ".venv" / "Scripts" / "python.exe"
    runner = ROOT / "scripts" / "w_manifest.py"
    try:
        subprocess.run(
            [str(py), str(runner), session_label], cwd=str(ROOT), check=False
        )
    except Exception as e:
        # Never fail the parent script because of manifest issues.
        print(f"[manifest] skipped ({e})")

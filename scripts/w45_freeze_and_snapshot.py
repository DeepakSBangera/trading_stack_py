from __future__ import annotations

import json
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(r"F:\Projects\trading_stack_py")
DOCS = ROOT / "docs"
REPORTS = ROOT / "reports"
CONFIG = ROOT / "config"

FREEZE_ZIP = REPORTS / f"W45_freeze_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.zip"

INCLUDE_FILES = [
    REPORTS / "run_manifest.jsonl",
    DOCS / "living_tracker.csv",
    CONFIG / "kill_switch.yaml",
]

INCLUDE_DIRS = [
    CONFIG,  # full config snapshot
    ROOT / "modules",  # source snapshot (lightweight)
    ROOT / "tradingstack",  # package snapshot
]


def _add_path(z: zipfile.ZipFile, path: Path, arcbase: str):
    if path.is_file():
        arcname = f"{arcbase}/{path.name}"
        z.write(path, arcname=arcname)
    elif path.is_dir():
        for p in path.rglob("*"):
            if p.is_file():
                rel = p.relative_to(path)
                z.write(p, arcname=f"{arcbase}/{path.name}/{rel.as_posix()}")


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(FREEZE_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for f in INCLUDE_FILES:
            if f.exists():
                _add_path(z, f, "files")
        for d in INCLUDE_DIRS:
            if d.exists():
                _add_path(z, d, "dirs")

    payload = {
        "created": str(FREEZE_ZIP),
        "bytes": FREEZE_ZIP.stat().st_size if FREEZE_ZIP.exists() else 0,
        "notes": "Production freeze snapshot: configs, run_manifest, tracker, modules, package.",
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

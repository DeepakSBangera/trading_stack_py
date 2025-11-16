# scripts/run_with_info.py
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r"F:\Projects\trading_stack_py")
REPORTS = ROOT / "reports"
CONFIG = ROOT / "config"
LOGDIR = REPORTS / "logs"
INFO_DIR = REPORTS / "script_info"

MAP_YAML = CONFIG / "script_info_map.yaml"


def safe_rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def snapshot_dir(dirpath: Path) -> dict[str, dict]:
    snap = {}
    if not dirpath.exists():
        return snap
    for p in dirpath.glob("*"):
        if p.is_file():
            st = p.stat()
            snap[str(p)] = {"size": st.st_size, "mtime": st.st_mtime}
    return snap


def detect_changes(before: dict, after: dict) -> list[Path]:
    out: list[Path] = []
    for k, v in after.items():
        if k not in before:
            out.append(Path(k))
        else:
            b = before[k]
            if v["size"] != b["size"] or v["mtime"] != b["mtime"]:
                out.append(Path(k))
    return out


def load_yaml(p: Path) -> dict:
    if not p.exists():
        return {}
    import yaml

    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def open_win(p: Path):
    if not p.exists():
        return
    try:
        subprocess.run(["cmd", "/c", "start", "", str(p)], check=False)
    except Exception:
        try:
            os.startfile(p)  # type: ignore[attr-defined]
        except Exception:
            print(f"[hint] open manually: {p}")


def write_script_card_inline(
    script_path: Path, purpose: str, session: str, outputs: list[Path], params: dict
) -> Path:
    INFO_DIR.mkdir(parents=True, exist_ok=True)
    name = script_path.name
    now = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    card = {
        "timestamp": now,
        "session": session,
        "script": safe_rel(script_path),
        "purpose": purpose,
        "inputs": [safe_rel(p) for p in params.get("inputs", [])],
        "outputs": [safe_rel(p) for p in outputs],
        "artifacts": [safe_rel(p) for p in outputs],
        "params": {k: v for k, v in params.items() if k != "inputs"},
        "log_file": params.get("log_file", ""),
        "return_code": params.get("return_code", None),
    }
    out = INFO_DIR / f"{name}.json"
    out.write_text(json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def write_script_card(
    script_path: Path, purpose: str, session: str, outputs: list[Path], params: dict
) -> Path:
    # Prefer helper if available; else inline
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        from _include_script_info import write_script_info  # type: ignore

        return write_script_info(
            str(script_path),
            purpose=purpose,
            inputs=[safe_rel(p) for p in params.get("inputs", [])],
            outputs=[safe_rel(p) for p in outputs],
            artifacts=[safe_rel(p) for p in outputs],
            params={k: v for k, v in params.items() if k != "inputs"},
            session=session,
        )
    except Exception:
        return write_script_card_inline(script_path, purpose, session, outputs, params)


def call_manifest(session_label: str):
    """Fire-and-forget run manifest."""
    runner = ROOT / "scripts" / "w_manifest.py"
    py = ROOT / ".venv" / "Scripts" / "python.exe"
    if runner.exists():
        try:
            subprocess.run(
                [str(py), str(runner), session_label], cwd=str(ROOT), check=False
            )
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(
        description="Run a script, auto-log outputs, refresh registry, (optional) manifest."
    )
    parser.add_argument(
        "target", help="Path to script (e.g., scripts\\w11_build_targets.py)"
    )
    parser.add_argument("--session", default="", help="Session label (e.g., S-W11)")
    parser.add_argument(
        "--purpose", default="", help="Override purpose text (else pulled from map)"
    )
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        help="Extra input paths to record (repeatable)",
    )
    parser.add_argument(
        "--args",
        nargs=argparse.REMAINDER,
        help="Args to pass to target (after --args ...)",
    )
    parser.add_argument(
        "--open",
        choices=["none", "card", "registry", "both"],
        default=os.environ.get("RUN_INFO_OPEN", "registry"),
        help="What to auto-open after run",
    )
    parser.add_argument(
        "--manifest",
        choices=["on", "off"],
        default=os.environ.get("RUN_INFO_MANIFEST", "on"),
        help="Also append a run manifest snapshot (default on)",
    )
    args = parser.parse_args()

    py = str(ROOT / ".venv" / "Scripts" / "python.exe")
    target = (
        ROOT / args.target
        if not args.target.startswith(str(ROOT))
        else Path(args.target)
    )
    if not target.exists():
        raise SystemExit(f"Target not found: {target}")

    # Purpose map
    purpose = args.purpose or load_yaml(MAP_YAML).get("map", {}).get(target.name, "")

    REPORTS.mkdir(parents=True, exist_ok=True)
    LOGDIR.mkdir(parents=True, exist_ok=True)

    # Pre-run snapshots
    before_reports = snapshot_dir(REPORTS)
    before_configs = snapshot_dir(CONFIG)

    # Run target with log capture
    ts = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = LOGDIR / f"{target.stem}_{ts}.log"
    cmd = [py, str(target)]
    if args.args:
        if len(args.args) and args.args[0] == "--":
            args.args = args.args[1:]
        cmd += args.args
    with log_file.open("w", encoding="utf-8") as lf:
        lf.write(f"# CMD: {' '.join(cmd)}\n# CWD: {ROOT}\n\n")
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for line in proc.stdout:  # type: ignore[arg-type]
            lf.write(line)
            sys.stdout.write(line)
        rc = proc.wait()

    # Post-run snapshots
    after_reports = snapshot_dir(REPORTS)
    after_configs = snapshot_dir(CONFIG)

    # Detect changed/new outputs
    changed_reports = detect_changes(before_reports, after_reports)
    outputs_sorted = sorted(changed_reports, key=lambda p: p.name.lower())

    # Inputs list: user-declared + all current configs
    cfg_inputs = [Path(k) for k in after_configs.keys()]
    user_inputs = [ROOT / p for p in args.input] if args.input else []
    params = {
        "return_code": rc,
        "log_file": safe_rel(log_file),
        "inputs": [*user_inputs, *cfg_inputs],
    }

    # Write script card
    card_path = write_script_card(target, purpose, args.session, outputs_sorted, params)

    # Optional manifest snapshot
    if args.manifest == "on":
        call_manifest(args.session or "")

    # Consolidate registry/MD
    consolidator = ROOT / "scripts" / "script_info_consolidate.py"
    if consolidator.exists():
        subprocess.run([py, str(consolidator)], cwd=str(ROOT), check=False)

    time.sleep(0.25)

    # Auto-open
    reg_csv = REPORTS / "script_registry.csv"
    if args.open in ("card", "both"):
        open_win(card_path)
    if args.open in ("registry", "both"):
        open_win(reg_csv)

    sys.exit(rc)


if __name__ == "__main__":
    main()

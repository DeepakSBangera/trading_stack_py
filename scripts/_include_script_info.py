# 2) FULL FILE CONTENTS — paste all below
# scripts/_include_script_info.py
from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

ROOT = Path(r"F:\Projects\trading_stack_py")
REPORTS = ROOT / "reports"
INFO_DIR = REPORTS / "script_info"


def _sha256_file(p: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _git(cmd: list[str]) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", *cmd],
            cwd=str(ROOT),
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return out.strip()
    except Exception:
        return None


def _git_info() -> dict:
    return {
        "sha": _git(["rev-parse", "HEAD"]),
        "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "tag_describe": _git(["describe", "--always", "--dirty", "--tags"]),
        "status_short": _git(["status", "--porcelain"]),
    }


def write_script_info(
    script_path: str,
    *,
    purpose: str = "",
    inputs: Iterable[str] = (),
    outputs: Iterable[str] = (),
    params: Mapping[str, Any] | None = None,
    artifacts: Iterable[str] = (),
    notes: str = "",
    session: str = "",
) -> Path:
    """
    Call from the bottom of any script to persist a per-script JSON 'card' and keep the latest snapshot.
    Example:
        from _include_script_info import write_script_info
        write_script_info(__file__, purpose="Build targets", outputs=["reports\\wk11_blend_targets.csv"], artifacts=[...], session="S-W11")
    """
    INFO_DIR.mkdir(parents=True, exist_ok=True)

    sp = Path(script_path)
    name = sp.name
    now = dt.datetime.now().astimezone().isoformat(timespec="seconds")

    # Normalize paths → strings relative to ROOT when possible
    def _norm(x: str) -> str:
        try:
            p = Path(x)
            return (
                str(p) if not str(p).startswith(str(ROOT)) else str(p.relative_to(ROOT))
            )
        except Exception:
            return str(x)

    inputs_l = sorted({_norm(s) for s in inputs})
    outputs_l = sorted({_norm(s) for s in outputs})
    arts_l = sorted({_norm(s) for s in artifacts})

    # quick hashes for outputs if exist
    out_hashes = {}
    for s in outputs_l:
        p = ROOT / s
        if p.exists():
            try:
                out_hashes[s] = {"sha256": _sha256_file(p), "bytes": p.stat().st_size}
            except Exception as e:
                out_hashes[s] = {"sha256": None, "error": str(e)}

    payload = {
        "timestamp": now,
        "session": session,
        "script": (
            str(sp.relative_to(ROOT)) if str(sp).startswith(str(ROOT)) else str(sp)
        ),
        "purpose": purpose,
        "inputs": inputs_l,
        "outputs": outputs_l,
        "artifacts": arts_l,
        "params": dict(params or {}),
        "notes": notes,
        "git": _git_info(),
        "output_hashes": out_hashes,
    }

    out_path = INFO_DIR / f"{name}.json"  # latest snapshot per script
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return out_path

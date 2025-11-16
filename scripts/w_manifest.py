# 2) FULL FILE CONTENTS — paste all below
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r"F:\Projects\trading_stack_py")
CONFIG = ROOT / "config"
REPORTS = ROOT / "reports"
DOCS = ROOT / "docs"

OUT_JSONL = REPORTS / "run_manifest.jsonl"  # append-only audit log (atomic emulation)
OUT_LAST = REPORTS / "run_manifest_last.json"  # last run (pretty)
OUT_INDEX = REPORTS / "run_manifest_index.csv"  # quick CSV index (atomic rewrite)


def open_win(p: Path):
    try:
        if sys.platform.startswith("win") and p.exists():
            subprocess.run(["cmd", "/c", "start", "", str(p)], check=False)
    except Exception:
        try:
            if p.exists():
                os.startfile(p)  # type: ignore[attr-defined]
        except Exception:
            print(f"[hint] open manually: {p}")


def sha256_file(p: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def git_cmd(args: list[str]) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", *args],
            cwd=str(ROOT),
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
        return out.strip()
    except Exception:
        return None


def get_git_info():
    return {
        "sha": git_cmd(["rev-parse", "HEAD"]),
        "branch": git_cmd(["rev-parse", "--abbrev-ref", "HEAD"]),
        "status_short": git_cmd(["status", "--porcelain"]),
        "tag_describe": git_cmd(["describe", "--always", "--dirty", "--tags"]),
    }


def list_artifacts():
    REPORTS.mkdir(parents=True, exist_ok=True)
    keep_ext = {".csv", ".json", ".jsonl", ".zip", ".html"}
    items = []
    for p in REPORTS.glob("*"):
        if p.suffix.lower() in keep_ext and p.is_file():
            stat = p.stat()
            items.append(
                {
                    "name": p.name,
                    "path": str(p),
                    "bytes": stat.st_size,
                    "mtime": dt.datetime.fromtimestamp(stat.st_mtime).isoformat(),
                }
            )
    items.sort(key=lambda r: r["mtime"], reverse=True)
    return items


def list_configs_and_hashes():
    out = {}
    if CONFIG.exists():
        for p in sorted(CONFIG.glob("*.*")):
            if p.is_file():
                try:
                    out[p.name] = {
                        "path": str(p),
                        "sha256": sha256_file(p),
                        "bytes": p.stat().st_size,
                    }
                except Exception as e:
                    out[p.name] = {"path": str(p), "sha256": None, "error": str(e)}
    return out


# ---------- Windows-safe atomic writers ----------


def _atomic_write_bytes(path: Path, data: bytes, retries: int = 15, delay: float = 0.2):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    for _ in range(retries):
        try:
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(delay)
        except Exception:
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
            raise
    # Fallback if still locked
    pend = path.with_suffix(path.suffix + f".{int(time.time())}.pending")
    with open(pend, "wb") as f:
        f.write(data)
    print(f"[warn] {path.name} locked; wrote pending copy {pend.name}")


def _atomic_write_text(path: Path, text: str):
    _atomic_write_bytes(path, text.encode("utf-8"))


def _atomic_append_text(path: Path, line: str, retries: int = 15, delay: float = 0.2):
    """
    Emulate append safely on Windows: read existing (if any), write whole file to tmp, replace.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            existing = ""
            if path.exists():
                try:
                    existing = path.read_text(encoding="utf-8")
                except PermissionError:
                    # Cannot read because someone locks it; wait and retry
                    time.sleep(delay)
                    continue
            data = (
                existing
                + ("" if existing.endswith("\n") or existing == "" else "\n")
                + line
            )
            _atomic_write_text(path, data)
            return
        except PermissionError:
            time.sleep(delay)
        except Exception:
            if attempt == retries - 1:
                pend = path.with_suffix(path.suffix + f".{int(time.time())}.pending")
                _atomic_write_text(pend, line + "\n")
                print(f"[warn] append locked; wrote pending copy {pend.name}")


def main():
    session_label = sys.argv[1] if len(sys.argv) > 1 else ""
    now = dt.datetime.now().astimezone()
    ts = now.isoformat(timespec="seconds")

    git = get_git_info()
    cfg = list_configs_and_hashes()
    arts = list_artifacts()

    manifest = {
        "timestamp": ts,
        "session": session_label,
        "machine": os.environ.get("COMPUTERNAME") or "",
        "repo_root": str(ROOT),
        "git": git,
        "configs": cfg,
        "artifacts": arts,
    }

    REPORTS.mkdir(parents=True, exist_ok=True)

    # JSONL append (atomic emulation)
    line = json.dumps(manifest, ensure_ascii=False)
    _atomic_append_text(OUT_JSONL, line)

    # Pretty last
    _atomic_write_text(OUT_LAST, json.dumps(manifest, indent=2, ensure_ascii=False))

    # CSV index (atomic rewrite)
    header = "timestamp,session,branch,git_sha8,artifacts_count,configs_count\n"
    row = f"{ts},{session_label},{git.get('branch') or ''},{(git.get('sha') or '')[:8]},{len(arts)},{len(cfg)}\n"
    try:
        existing = (
            OUT_INDEX.read_text(encoding="utf-8") if OUT_INDEX.exists() else header
        )
        if not existing.startswith("timestamp,"):
            existing = header  # sanitize bad header
    except PermissionError:
        existing = header  # if locked for read, we’ll just rewrite with header + row
    text = existing.strip() + ("\n" if not existing.endswith("\n") else "") + row
    _atomic_write_text(OUT_INDEX, text)

    print(
        json.dumps(
            {
                "jsonl": str(OUT_JSONL),
                "last_json": str(OUT_LAST),
                "index_csv": str(OUT_INDEX),
                "artifacts_count": len(arts),
                "configs_count": len(cfg),
                "git_sha8": (git.get("sha") or "")[:8],
            },
            indent=2,
        )
    )

    # Optional: open views (commented to avoid locks)
    # open_win(OUT_LAST); open_win(OUT_INDEX)


if __name__ == "__main__":
    main()

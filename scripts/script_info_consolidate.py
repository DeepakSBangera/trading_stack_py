# 2) FULL FILE CONTENTS — paste all below
# scripts/script_info_consolidate.py
from __future__ import annotations

import io
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(r"F:\Projects\trading_stack_py")
REPORTS = ROOT / "reports"
DOCS = ROOT / "docs"
INFO_DIR = REPORTS / "script_info"
OUT_REG = REPORTS / "script_registry.csv"
OUT_MD = DOCS / "SCRIPTS_LIVING.md"


def open_win(p: Path):
    if sys.platform.startswith("win") and p.exists():
        try:
            import subprocess

            subprocess.run(["cmd", "/c", "start", "", str(p)], check=False)
        except Exception:
            try:
                os.startfile(p)  # type: ignore[attr-defined]
            except Exception:
                print(f"[hint] open manually: {p}")


def _atomic_write_bytes(path: Path, data: bytes, retries: int = 15, delay: float = 0.2):
    """
    Windows-safe write: write to .tmp then os.replace(.tmp, target).
    Retries if the target is locked by another process (Excel/Notepad).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    for attempt in range(retries):
        try:
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, path)
            return
        except PermissionError:
            # someone has the file open; wait and retry
            time.sleep(delay)
        except Exception:
            # best effort cleanup of tmp
            try:
                if tmp.exists():
                    tmp.unlink(missing_ok=True)  # type: ignore[arg-type]
            except Exception:
                pass
            raise
    # last attempt: write a side file so work isn't lost
    fallback = path.with_suffix(path.suffix + f".{int(time.time())}.pending")
    with open(fallback, "wb") as f:
        f.write(data)
    print(f"[warn] Could not replace {path.name} (locked). Wrote pending copy: {fallback.name}")


def _atomic_write_text(path: Path, text: str):
    _atomic_write_bytes(path, text.encode("utf-8"))


def load_cards() -> pd.DataFrame:
    rows = []
    if not INFO_DIR.exists():
        return pd.DataFrame()
    for p in sorted(INFO_DIR.glob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            rows.append(
                {
                    "script": d.get("script", p.name.replace(".json", "")),
                    "session": d.get("session", ""),
                    "purpose": d.get("purpose", ""),
                    "timestamp": d.get("timestamp", ""),
                    "inputs": "; ".join(d.get("inputs", [])),
                    "outputs": "; ".join(d.get("outputs", [])),
                    "artifacts": "; ".join(d.get("artifacts", [])),
                    "params": json.dumps(d.get("params", {}), ensure_ascii=False),
                    "git_sha8": (d.get("git", {}).get("sha") or "")[:8],
                }
            )
        except Exception as e:
            rows.append({"script": p.name, "purpose": f"ERROR {e}"})
    return pd.DataFrame(rows)


def write_markdown(df: pd.DataFrame):
    DOCS.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# Scripts Living Document\n")
    lines.append("_Latest snapshot for each script — auto-generated._\n")
    if df.empty:
        lines.append("\n_No script info cards found._\n")
    else:
        lines.append("\n## Registry (latest per script)\n")
        lines.append("| Script | Session | Last Run (IST) | Purpose | Outputs | Artifacts | git |\n")
        lines.append("|---|---|---|---|---|---|---|\n")
        for _, r in df.sort_values("script").iterrows():
            lines.append(
                f"| `{r['script']}` | {r['session']} | {r['timestamp']} | {r['purpose']} | {r['outputs']} | {r['artifacts']} | {r['git_sha8']} |\n"
            )
        lines.append("\n---\n")
        lines.append("## Details by Script\n")
        for _, r in df.sort_values("script").iterrows():
            lines.append(f"\n### `{r['script']}`\n")
            lines.append(f"- **Session**: {r['session']}\n")
            lines.append(f"- **Last run**: {r['timestamp']}\n")
            lines.append(f"- **Purpose**: {r['purpose']}\n")
            lines.append(f"- **Inputs**: {r['inputs']}\n")
            lines.append(f"- **Outputs**: {r['outputs']}\n")
            lines.append(f"- **Artifacts**: {r['artifacts']}\n")
            lines.append(f"- **Params**: `{r['params']}`\n")
            lines.append(f"- **git**: {r['git_sha8']}\n")
    _atomic_write_text(OUT_MD, "\n".join(lines))


def main():
    df = load_cards()
    REPORTS.mkdir(parents=True, exist_ok=True)
    # Write CSV via in-memory buffer → atomic replace
    if df.empty:
        df = pd.DataFrame(
            columns=[
                "script",
                "session",
                "purpose",
                "timestamp",
                "inputs",
                "outputs",
                "artifacts",
                "params",
                "git_sha8",
            ]
        )
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    _atomic_write_text(OUT_REG, buf.getvalue())
    write_markdown(df)
    print(
        {
            "registry_csv": str(OUT_REG),
            "living_md": str(OUT_MD),
            "rows": int(df.shape[0]),
        }
    )
    # Don’t auto-open to avoid re-locking; wrapper decides what to open.


if __name__ == "__main__":
    main()

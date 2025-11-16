from __future__ import annotations

import datetime as _dt
import html
import pathlib

# --- repo-import shim: allow local package imports without install
import sys  # noqa: E401
from pathlib import Path

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # repo root

REPORTS = Path("reports")

_RAW_HTML = r"""
<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>Report Index</title>
    <style>
      body { font-family: Arial, Helvetica, sans-serif; margin: 24px; }
      h1   { margin-bottom: 12px; }
      .grid { display: grid; grid-template-columns: 1fr 140px 200px; gap: 8px 16px; }
      .hdr  { font-weight: bold; border-bottom: 1px solid #ddd; padding-bottom: 6px; }
      .row  { padding: 4px 0; border-bottom: 1px dashed #eee; }
      a     { text-decoration: none; }
      .muted { color: #666; font-size: 12px; }
    </style>
  </head>
  <body>
    <h1>Reports</h1>
    <div class="muted">Generated at {generated_at}</div>
    <div class="grid" style="margin-top:12px">
      <div class="hdr">File</div><div class="hdr">Size</div><div class="hdr">Last Write</div>
      {rows}
    </div>
  </body>
</html>
"""

# Escape non-placeholder braces so str.format() doesn't choke on CSS
HTML = (
    _RAW_HTML.replace("{", "{{")
    .replace("}", "}}")
    .replace("{{generated_at}}", "{generated_at}")
    .replace("{{rows}}", "{rows}")
)


def _fmt_bytes(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024 or unit == "GB":
            return (
                f"{n:.0f} {unit}"
                if unit == "B"
                else (
                    f"{n / 1024:.1f} {unit}" if unit in ("KB", "MB", "GB") else f"{n} B"
                )
            )
        n /= 1024
    return f"{n:.0f} B"


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    files = []
    for p in REPORTS.rglob("*"):
        if not p.is_file():
            continue
        # Skip huge binaries to keep the page snappy
        if p.suffix.lower() in {".zip", ".pkl"}:
            continue
        rel = p.relative_to(REPORTS).as_posix()
        stat = p.stat()
        files.append((rel, stat.st_size, stat.st_mtime))

    files.sort(key=lambda x: x[0].lower())

    rows = []
    for rel, size, mtime in files:
        href = html.escape(rel)
        name = html.escape(rel)
        size_s = _fmt_bytes(size)
        ts = _dt.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
        rows.append(
            f'<div class="row"><div><a href="{href}">{name}</a></div>'
            f"<div>{size_s}</div><div>{ts}</div></div>"
        )

    html_out = HTML.format(
        generated_at=_dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        rows=(
            "\n".join(rows)
            if rows
            else '<div class="row"><div>(no files)</div><div></div><div></div></div>'
        ),
    )
    (REPORTS / "index.html").write_text(html_out, encoding="utf-8")
    print(f"[OK] Wrote: {REPORTS / 'index.html'}")


if __name__ == "__main__":
    main()

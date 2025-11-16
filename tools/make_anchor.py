#!/usr/bin/env python
"""
make_anchor.py — build/append a living scripts anchor.

Usage:
  python scripts/tools/make_anchor.py --anchor docs/ANCHOR_SCRIPTS.md [--section "W3 — Costs & Churn"] <files...>

What it does:
- Creates the anchor file if missing (with a header & TOC).
- For each provided script, extracts:
  • first docstring / top comments,
  • CLI (argparse) flags if present,
  • top-level functions/classes (AST),
  • obvious inputs/outputs mentioned in strings/paths,
  • a short purpose line (fallback = filename).
- Writes/updates a **Section** (e.g., W3) containing:
  • a compact inventory table,
  • per-script detail blocks.
- Idempotent: re-runs replace the same section.
"""
from __future__ import annotations

import argparse, ast, re, sys, textwrap, pathlib
from typing import List, Tuple

ANCHOR_HEADER = """# Scripts Anchor (Living)

**Purpose.** Single source of truth mapping each script → purpose, inputs/outputs, flags, & dependencies.  
**How to update.** Run the generator with the touched files; it replaces/creates the relevant section in-place.

"""

SECTION_TPL = """
## {section}

### Inventory (quick view)

| File | Purpose (1-line) | Inputs | Outputs | CLI Flags |
|---|---|---|---|---|
{table_rows}

{details}
"""

DETAIL_TPL = """
### `{name}`
**Path:** `{path}`  
**Purpose:** {purpose}

**Inputs (detected):** {inputs}  
**Outputs (detected):** {outputs}  
**CLI Flags (argparse):** {flags}

**API Surface (top-level):** {api}

**Notes:** {notes}
"""


def read_text(p: pathlib.Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def first_docstring_or_header(src: str) -> str:
    # Try AST docstring first
    try:
        tree = ast.parse(src)
        ds = ast.get_docstring(tree)
        if ds:
            return ds.strip().splitlines()[0][:200]
    except Exception:
        pass
    # Fallback: top # comment block
    m = re.match(r"(?:\s*#.*\n)+", src)
    if m:
        line = [ln.strip("# ").strip() for ln in m.group(0).splitlines() if ln.strip().startswith("#")]
        if line:
            return line[0][:200]
    return ""


def find_argparse_flags(src: str) -> List[str]:
    flags = []
    for m in re.finditer(r"add_argument\((?P<q>['\"]).*?(?P=q).*?\)", src, flags=re.S):
        chunk = m.group(0)
        picks = re.findall(r"['\"](--[a-zA-Z0-9][a-zA-Z0-9\-_]*)['\"]", chunk)
        flags.extend(picks)
    return sorted(set(flags))


def find_paths_io(src: str) -> Tuple[List[str], List[str]]:
    # naive IO detection by common verbs & csv/parquet/png/json
    inputs, outputs = set(), set()
    for pat in [r"read_csv\(['\"]([^'\"]+)", r"read_parquet\(['\"]([^'\"]+)", r"open\(['\"]([^'\"]+)['\"],\s*['\"]r"]:
        inputs.update(re.findall(pat, src))
    for pat in [
        r"to_csv\(['\"]([^'\"]+)",
        r"to_parquet\(['\"]([^'\"]+)",
        r"savefig\(['\"]([^'\"]+)",
        r"open\(['\"]([^'\"]+)['\"],\s*['\"]w",
    ]:
        outputs.update(re.findall(pat, src))
    return sorted(inputs), sorted(outputs)


def top_level_api(src: str) -> List[str]:
    items = []
    try:
        tree = ast.parse(src)
        for n in tree.body:
            if isinstance(n, ast.FunctionDef):
                items.append(f"def {n.name}({', '.join(a.arg for a in n.args.args)})")
            elif isinstance(n, ast.ClassDef):
                items.append(f"class {n.name}")
    except Exception:
        pass
    return items[:12]


def purpose_guess(name: str, docline: str) -> str:
    if docline:
        return docline
    base = name.replace("_", " ").replace(".py", "")
    return f"{base} — utility script"


def render_table_row(md_name: str, purpose: str, ins: List[str], outs: List[str], flags: List[str]) -> str:
    ins_s = (", ".join(ins) or "—")[:60]
    outs_s = (", ".join(outs) or "—")[:60]
    fl_s = (" ".join(flags) or "—")[:60]
    return f"| `{md_name}` | {purpose} | {ins_s} | {outs_s} | {fl_s} |"


def replace_or_insert_section(anchor_text: str, section: str, payload: str) -> str:
    # Replace between headings "## {section}" and next "## "
    pattern = rf"(?:^|\n)## {re.escape(section)}\n.*?(?=\n## |\Z)"
    if re.search(pattern, anchor_text, flags=re.S):
        return re.sub(pattern, "\n" + payload.strip(), anchor_text, flags=re.S)
    else:
        return anchor_text.rstrip() + "\n\n" + payload.strip() + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", default="docs/ANCHOR_SCRIPTS.md")
    ap.add_argument("--section", default="Unlabeled")
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()

    anchor_path = pathlib.Path(args.anchor)
    anchor_text = anchor_path.read_text(encoding="utf-8") if anchor_path.exists() else ANCHOR_HEADER

    rows = []
    detail_blocks = []
    for f in args.files:
        p = pathlib.Path(f)
        src = read_text(p)
        doc = first_docstring_or_header(src)
        flags = find_argparse_flags(src)
        ins, outs = find_paths_io(src)
        api = top_level_api(src)

        rows.append(render_table_row(p.name, purpose_guess(p.name, doc), ins, outs, flags))

        detail_blocks.append(
            DETAIL_TPL.format(
                name=p.name,
                path=str(p),
                purpose=purpose_guess(p.name, doc),
                inputs=", ".join(ins) if ins else "—",
                outputs=", ".join(outs) if outs else "—",
                flags=" ".join(flags) if flags else "—",
                api=", ".join(api) if api else "—",
                notes="(auto-generated; refine this notes line if needed)",
            ).strip()
        )

    payload = SECTION_TPL.format(
        section=args.section, table_rows="\n".join(rows), details="\n\n".join(detail_blocks)
    ).strip()

    updated = replace_or_insert_section(anchor_text, args.section, payload)
    anchor_path.parent.mkdir(parents=True, exist_ok=True)
    anchor_path.write_text(updated, encoding="utf-8")

    # stdout hint for your logs
    print(f"[anchor] Updated: {anchor_path}  (section: {args.section}; files: {len(rows)})")


if __name__ == "__main__":
    main()

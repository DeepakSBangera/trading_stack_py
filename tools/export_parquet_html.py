from __future__ import annotations

import argparse
from html import escape
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser(
        description="Export a parquet file to HTML table (lightweight viewer)."
    )
    ap.add_argument("parquet", nargs="?", default="reports/factor_exposures.parquet")
    ap.add_argument(
        "--out", default=None, help="Output HTML path (default: alongside parquet)"
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Max rows to render (0 or negative = all)",
    )
    ap.add_argument("--title", default="Parquet Preview", help="HTML title")
    args = ap.parse_args()

    p = Path(args.parquet)
    if not p.exists():
        raise SystemExit(f"Input parquet not found: {p}")

    df = pd.read_parquet(p)
    total = len(df)

    if args.limit and args.limit > 0 and total > args.limit:
        df = df.head(args.limit)
        note = f"Showing first {len(df):,} of {total:,} rows"
    else:
        note = f"Showing all {total:,} rows"

    # Render
    table_html = df.to_html(border=1, classes="dataframe", justify="center")
    head = f"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{escape(args.title)}</title>
<style>
body {{ font-family: Segoe UI, Roboto, Arial, sans-serif; margin: 20px; }}
h1 {{ font-size: 18px; margin: 0 0 8px 0; }}
.note {{ color: #555; margin-bottom: 12px; }}
.dataframe {{ border-collapse: collapse; font-size: 12px; }}
.dataframe th, .dataframe td {{ padding: 4px 8px; }}
.dataframe tr:nth-child(even) {{ background: #fafafa; }}
</style>
</head>
<body>
<h1>{escape(args.title)}</h1>
<div class="note">{escape(note)}</div>
{table_html}
</body>
</html>
"""

    out = Path(args.out) if args.out else p.with_suffix(".html")
    out.write_text(head, encoding="utf-8")
    print(f"[OK] Wrote HTML: {out.resolve()}")


if __name__ == "__main__":
    main()

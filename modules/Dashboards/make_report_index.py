from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Daily Report Index</title>
<style>
  body { font-family: Segoe UI, Roboto, Arial, sans-serif; margin: 24px; }
  h1 { margin: 0 0 12px 0; font-size: 20px; }
  .small { color: #666; font-size: 12px; margin-bottom: 16px; }
  .grid { display: grid; grid-template-columns: repeat(2, minmax(280px, 1fr)); gap: 14px; }
  .card { border: 1px solid #ddd; border-radius: 8px; padding: 12px; }
  img { max-width: 100%; height: auto; border: 1px solid #eee; }
  a { text-decoration: none; color: #0b5ed7; }
</style>
</head>
<body>
<h1>Daily Report Index</h1>
<div class="small">Generated: {now}</div>

<div class="grid">
  <div class="card">
    <h2 style="margin:0 0 8px 0;font-size:16px;">Tearsheet v2</h2>
    <div>{tearsheet_links}</div>
    {tearsheet_img}
  </div>

  <div class="card">
    <h2 style="margin:0 0 8px 0;font-size:16px;">Artifacts</h2>
    <ul>
      <li><a href="rolling_metrics.parquet">rolling_metrics.parquet</a></li>
      <li><a href="rolling_metrics_summary.txt">rolling_metrics_summary.txt</a></li>
      <li><a href="factor_exposures.parquet">factor_exposures.parquet</a></li>
      <li><a href="factor_exposures_summary.txt">factor_exposures_summary.txt</a></li>
      <li><a href="portfolio_v2.parquet">portfolio_v2.parquet</a></li>
    </ul>
  </div>
</div>

</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Write a simple reports/index_daily.html")
    ap.add_argument("--outdir", default="reports")
    ap.add_argument("--tearsheet-png", default="reports/tearsheet_v2.png")
    ap.add_argument("--tearsheet-html", default="reports/tearsheet_v2.html")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    png_path = Path(args.tearsheet_png)
    html_path = Path(args.tearsheet_html)

    # Links (show only what exists)
    links: list[str] = []
    if html_path.exists():
        rel = html_path.name if html_path.parent == outdir else str(html_path)
        links.append(f'<a href="{rel}">Open interactive HTML</a>')
    if png_path.exists():
        rel = png_path.name if png_path.parent == outdir else str(png_path)
        links.append(f'<a href="{rel}" style="margin-left:10px;">Open PNG</a>')
    tearsheet_links = " | ".join(links) if links else "<em>No tearsheet found</em>"

    # Inline preview if PNG present
    if png_path.exists():
        rel_img = png_path.name if png_path.parent == outdir else str(png_path)
        tearsheet_img = f'<div style="margin-top:10px;"><img src="{rel_img}" alt="Tearsheet v2" /></div>'
    else:
        tearsheet_img = ""

    html = HTML.format(
        now=datetime.now().strftime("%Y-%m-%d %H:%M"),
        tearsheet_links=tearsheet_links,
        tearsheet_img=tearsheet_img,
    )

    (outdir / "index_daily.html").write_text(html, encoding="utf-8")
    print(f"[OK] Wrote: {outdir / 'index_daily.html'}")


if __name__ == "__main__":
    main()

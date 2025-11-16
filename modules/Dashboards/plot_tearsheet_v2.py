from __future__ import annotations

import html
from pathlib import Path

import numpy as np
import pandas as pd


def _fmt_pct(x: float | None) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return f"{x:.2%}"


def render_tearsheet_v2(
    last_date: pd.Timestamp | None,
    last_nav: float,
    roll_sh_last: float | None,
    roll_vol_last: float | None,
    roll_mdd_last: float | None,
    regime_last: float | None,
    ann_vol: float,
    mom_last: float | None,
    qual_last: float | None,
    max_dd: float,
) -> str:
    """Return an HTML string for the Tearsheet v2 dashboard (all lines <= 88 cols)."""

    # Header + styles (split across short lines)
    head_lines = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'/>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'/>",
        "<style>",
        "body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;",
        "       margin: 20px; color:#111; }",
        "h1 { font-size: 18px; margin: 0 0 6px 0; }",
        "h2 { font-size: 16px; margin: 24px 0 8px; }",
        ".metric-grid { display: grid;",
        "  grid-template-columns: repeat(3, minmax(200px,1fr));",
        "  gap:10px; margin-top:10px; }",
        ".card { border:1px solid #ddd; border-radius:8px; padding:10px; }",
        ".small { color:#666; font-size:12px }",
        "</style></head><body>",
        "<h1>Tearsheet v2</h1>",
    ]

    # Safe text for "last date"
    last_date_txt = "NA"
    if last_date is not None and pd.notnull(last_date):
        try:
            last_date_txt = str(pd.Timestamp(last_date).date())
        except Exception:
            last_date_txt = "NA"

    head_lines.append(f"<div class='small'>Last date: {html.escape(last_date_txt)}</div>")

    # KPI cards (each value rendered safely; empty string if missing)
    def _num_or_empty(x: float | None, fmt: str) -> str:
        if x is None:
            return ""
        try:
            if isinstance(x, (int, float)) and not np.isnan(x):
                return fmt.format(x)
        except Exception:
            pass
        return ""

    nav_txt = _num_or_empty(last_nav, "{:.4f}")
    sh_txt = _num_or_empty(roll_sh_last, "{:.2f}")
    vol_txt = _num_or_empty(roll_vol_last, "{:.2f}")
    mdd_txt = "" if roll_mdd_last is None else _fmt_pct(roll_mdd_last)
    reg_txt = ""
    if regime_last is not None:
        try:
            reg_txt = str(int(regime_last))
        except Exception:
            reg_txt = ""

    ann_vol_txt = "" if np.isnan(ann_vol) else f"{ann_vol:.2%}"
    mom_txt = "" if mom_last is None else f"{mom_last:.2%}"
    qlt_txt = "" if qual_last is None else f"{qual_last:.3f}"
    max_dd_txt = "" if np.isnan(max_dd) else f"{max_dd:.2%}"

    kpi_lines = [
        "<div class='metric-grid'>",
        ("<div class='card'><div class='small'>Last NAV</div>" f"<div style='font-size:22px'>{nav_txt}</div></div>"),
        (
            "<div class='card'><div class='small'>Rolling Sharpe (last)</div>"
            f"<div style='font-size:22px'>{sh_txt}</div></div>"
        ),
        (
            "<div class='card'><div class='small'>Rolling Vol (last)</div>"
            f"<div style='font-size:22px'>{vol_txt}</div></div>"
        ),
        (
            "<div class='card'><div class='small'>Rolling Max DD (last)</div>"
            f"<div style='font-size:22px'>{mdd_txt}</div></div>"
        ),
        (
            "<div class='card'><div class='small'>Regime (last)</div>"
            f"<div style='font-size:22px'>{reg_txt}</div></div>"
        ),
        (
            "<div class='card'><div class='small'>Period Ann. Vol</div>"
            f"<div style='font-size:22px'>{ann_vol_txt}</div></div>"
        ),
        (
            "<div class='card'><div class='small'>Momentum 12-1 (last)</div>"
            f"<div style='font-size:22px'>{mom_txt}</div></div>"
        ),
        (
            "<div class='card'><div class='small'>Quality Inv Down Vol (last)</div>"
            f"<div style='font-size:22px'>{qlt_txt}</div></div>"
        ),
        (
            "<div class='card'><div class='small'>Max Drawdown (period)</div>"
            f"<div style='font-size:22px'>{max_dd_txt}</div></div>"
        ),
        "</div>",
    ]

    body_lines = ["<h2>Overview</h2>"]
    # The rest of the overview (tables/plots) can be appended by caller.

    tail_lines = ["</body></html>"]

    html_out = "\n".join(head_lines + kpi_lines + body_lines + tail_lines)
    return html_out


def write_tearsheet_file(
    out_path: Path,
    last_date: pd.Timestamp | None,
    last_nav: float,
    roll_sh_last: float | None,
    roll_vol_last: float | None,
    roll_mdd_last: float | None,
    regime_last: float | None,
    ann_vol: float,
    mom_last: float | None,
    qual_last: float | None,
    max_dd: float,
) -> Path:
    """Write the HTML to out_path."""
    out_html = render_tearsheet_v2(
        last_date=last_date,
        last_nav=last_nav,
        roll_sh_last=roll_sh_last,
        roll_vol_last=roll_vol_last,
        roll_mdd_last=roll_mdd_last,
        regime_last=regime_last,
        ann_vol=ann_vol,
        mom_last=mom_last,
        qual_last=qual_last,
        max_dd=max_dd,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(out_html, encoding="utf-8")
    return out_path

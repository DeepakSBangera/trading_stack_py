from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tradingstack.utils.dates import coerce_date_index


# ----------------- loaders -----------------
def _load_parquet_with_date_index(p: Path) -> pd.DataFrame:
    df = pd.read_parquet(p)
    return coerce_date_index(df, date_col="date").sort_index()


def _load_portfolio(p: Path) -> tuple[pd.DataFrame, str | None]:
    df = pd.read_parquet(p)
    df = coerce_date_index(df, date_col="date")

    nav_col = next(
        (c for c in ("nav_net", "nav_gross", "_nav", "nav") if c in df.columns), None
    )

    # returns
    if "ret_net" in df.columns:
        df["returns"] = pd.to_numeric(df["ret_net"], errors="coerce")
    elif "ret_gross" in df.columns:
        df["returns"] = pd.to_numeric(df["ret_gross"], errors="coerce")
    else:
        if nav_col:
            df["returns"] = pd.to_numeric(df[nav_col], errors="coerce").pct_change()
        else:
            df["returns"] = np.nan

    return df, nav_col


# ----------------- metrics helpers -----------------
def _annualize_vol(returns: pd.Series, trading_days: int = 252) -> float:
    d = pd.to_numeric(returns, errors="coerce").dropna()
    if d.empty:
        return float("nan")
    return float(d.std(ddof=0) * np.sqrt(252.0))


def _max_drawdown(nav: pd.Series) -> float:
    x = pd.to_numeric(nav, errors="coerce").dropna()
    if x.empty:
        return float("nan")
    dd = (x / x.cummax()) - 1.0
    return float(dd.min())


def _last_nonnull(s: pd.Series | None) -> float | None:
    if s is None:
        return None
    try:
        return float(pd.to_numeric(s, errors="coerce").dropna().iloc[-1])
    except Exception:
        return None


def _plot_series_or_note(ax, idx, series: pd.Series | None, title: str):
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if series is None:
        ax.text(0.5, 0.5, "not found", ha="center", va="center", transform=ax.transAxes)
        return
    s = pd.to_numeric(series, errors="coerce")
    n_valid = int(s.count())
    if n_valid == 0:
        ax.text(
            0.5,
            0.5,
            "no valid points",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return
    ax.plot(idx, s)


# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser(description="Generate Tearsheet v2 (PNG + HTML)")
    ap.add_argument("--portfolio", default="reports/portfolio_v2.parquet")
    ap.add_argument("--rolling", default="reports/rolling_metrics.parquet")
    ap.add_argument("--factors", default="reports/factor_exposures.parquet")
    ap.add_argument("--outdir", default="reports")
    ap.add_argument("--png", default="tearsheet_v2.png")
    ap.add_argument("--html", default="tearsheet_v2.html")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    missing: list[str] = []

    # portfolio
    try:
        port, nav_col = _load_portfolio(Path(args.portfolio))
    except Exception as e:
        raise SystemExit(f"Cannot build tearsheet: portfolio error: {e}") from e
    if port.empty:
        raise SystemExit("Cannot build tearsheet: portfolio data unavailable.")

    # rolling (best-effort)
    try:
        roll = _load_parquet_with_date_index(Path(args.rolling))
    except Exception as e:
        missing.append(f"rolling metrics ({e})")
        roll = pd.DataFrame(index=port.index)

    # factors (best-effort)
    try:
        fac = _load_parquet_with_date_index(Path(args.factors))
    except Exception as e:
        missing.append(f"factor exposures ({e})")
        fac = pd.DataFrame(index=port.index)

    # -------- figure --------
    plt.close("all")
    fig = plt.figure(figsize=(14, 10), dpi=150)
    gs = fig.add_gridspec(
        3,
        2,
        height_ratios=[1.2, 1.0, 1.0],
        width_ratios=[1.0, 1.0],
        hspace=0.30,
        wspace=0.20,
    )

    # Equity curve
    ax1 = fig.add_subplot(gs[0, :])
    if nav_col:
        ax1.plot(port.index, port[nav_col], linewidth=1.5)
        ax1.set_title(f"Equity Curve ({nav_col})")
    else:
        eq = (
            1.0 + pd.to_numeric(port["returns"], errors="coerce").fillna(0.0)
        ).cumprod()
        ax1.plot(eq.index, eq.values, linewidth=1.5)
        ax1.set_title("Equity Curve (synthetic from returns)")
    ax1.grid(True, alpha=0.3)

    # Rolling panels
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[2, 0])

    r_idx = roll.index if not roll.empty else port.index
    _plot_series_or_note(ax2, r_idx, roll.get("rolling_sharpe"), "Rolling Sharpe")
    _plot_series_or_note(ax3, r_idx, roll.get("rolling_vol"), "Rolling Volatility")
    _plot_series_or_note(ax4, r_idx, roll.get("rolling_mdd"), "Rolling Max Drawdown")

    # Factors panel
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.grid(True, alpha=0.3)
    if fac.empty or fac.shape[1] == 0:
        ax5.set_title("Factors — not available")
    else:
        drew_any = False
        if "mom_12_1_proxy" in fac.columns:
            ax5.plot(
                fac.index,
                pd.to_numeric(fac["mom_12_1_proxy"], errors="coerce"),
                label="Momentum 12-1 Proxy",
                linewidth=1.0,
            )
            drew_any = True
        if "quality_inv_downside_vol" in fac.columns:
            ax5.plot(
                fac.index,
                pd.to_numeric(fac["quality_inv_downside_vol"], errors="coerce"),
                label="Quality (Inv Down Vol)",
                linewidth=1.0,
            )
            drew_any = True
        sector_cols = [c for c in fac.columns if c.startswith("sector_")]
        if sector_cols:
            fac_tail = fac[sector_cols].tail(252)
            if not fac_tail.empty:
                rowsum = fac_tail.sum(axis=1).replace(0.0, np.nan)
                fac_tail = fac_tail.div(rowsum, axis=0).fillna(0.0)
                ax_inset = ax5.inset_axes([0.05, 0.05, 0.9, 0.45])
                ax_inset.stackplot(fac_tail.index, fac_tail.values.T)
                ax_inset.set_title("Sector exposures (last ~252 days)", fontsize=9)
                ax_inset.grid(True, alpha=0.2)
                drew_any = True
        ax5.set_title("Factors" if drew_any else "Factors — no valid points")
        if drew_any:
            ax5.legend(loc="best", fontsize=9)

    fig.suptitle("Tearsheet v2", fontsize=14, y=0.98)
    png_path = outdir / args.png
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)

    # ---- HTML summary ----
    last_date = port.index.max()
    nav_series = (
        pd.to_numeric(port[nav_col], errors="coerce")
        if nav_col
        else (1.0 + port["returns"].fillna(0.0)).cumprod()
    )
    last_nav = float(nav_series.dropna().iloc[-1])
    ann_vol = _annualize_vol(port["returns"])
    max_dd = _max_drawdown(nav_series)

    def _get(col):
        return fac.get(col) if (not fac.empty and col in fac.columns) else None

    roll_vol_last = _last_nonnull(roll.get("rolling_vol"))
    roll_sh_last = _last_nonnull(roll.get("rolling_sharpe"))
    roll_mdd_last = _last_nonnull(roll.get("rolling_mdd"))
    regime_last = _last_nonnull(roll.get("regime"))
    mom_last = _last_nonnull(_get("mom_12_1_proxy"))
    qual_last = _last_nonnull(_get("quality_inv_downside_vol"))

    sector_cols = [
        c for c in (fac.columns if not fac.empty else []) if c.startswith("sector_")
    ]
    top_sector_html = ""
    if sector_cols:
        last_sectors = fac[sector_cols].dropna(how="all").tail(1).T
        if not last_sectors.empty:
            last_sectors.columns = ["weight"]
            last_sectors = last_sectors.sort_values(
                "weight", key=lambda s: s.abs(), ascending=False
            ).head(6)
            top_sector_html = last_sectors.to_html(
                border=1, classes="dataframe", float_format=lambda x: f"{x:.3f}"
            )

    missing_note = (
        f"<div class='note'>Missing inputs: {', '.join(missing)}</div>"
        if missing
        else ""
    )

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Tearsheet v2</title>
<style>
body {{ font-family: Segoe UI, Roboto, Arial, sans-serif; margin: 24px; }}
h1 {{ font-size: 18px; margin: 0 0 6px 0; }}
h2 {{ font-size: 16px; margin: 24px 0 8px; }}
.metric-grid {{ display:grid; grid-template-columns: repeat(3, minmax(200px,1fr)); gap:10px; margin-top:10px; }}
.card {{ border:1px solid #ddd; border-radius:8px; padding:10px; }}
.small {{ color:#666; font-size:12px }}
img {{ max-width:100%; height:auto; border:1px solid #eee; }}
.note {{ color:#aa0000; margin-top:6px; }}
</style></head><body>
<h1>Tearsheet v2</h1>
<div class="small">Last date: {last_date.date() if pd.notnull(last_date) else 'NA'}</div>
<div class="metric-grid">
  <div class="card"><div class="small">Last NAV</div><div style="font-size:22px">{last_nav:.4f}</div></div>
  <div class="card"><div class="small">Rolling Sharpe (last)</div><div style="font-size:22px">{'' if roll_sh_last is None else f'{roll_sh_last:.2f}'}</div></div>
  <div class="card"><div class="small">Rolling Vol (last)</div><div style="font-size:22px">{'' if roll_vol_last is None else f'{roll_vol_last:.2f}'}</div></div>
  <div class="card"><div class="small">Rolling Max DD (last)</div><div style="font-size:22px">{'' if roll_mdd_last is None else f'{roll_mdd_last:.2%}'}</div></div>
  <div class="card"><div class="small">Regime (last)</div><div style="font-size:22px">{'' if regime_last is None else int(regime_last)}</div></div>
  <div class="card"><div class="small">Period Ann. Vol</div><div style="font-size:22px">{'' if np.isnan(ann_vol) else f'{ann_vol:.2%}'}</div></div>
  <div class="card"><div class="small">Momentum 12-1 (last)</div><div style="font-size:22px">{'' if mom_last is None else f'{mom_last:.2%}'}</div></div>
  <div class="card"><div class="small">Quality Inv Down Vol (last)</div><div style="font-size:22px">{'' if qual_last is None else f'{qual_last:.3f}'}</div></div>
  <div class="card"><div class="small">Max Drawdown (period)</div><div style="font-size:22px">{'' if np.isnan(max_dd) else f'{max_dd:.2%}'}</div></div>
</div>
<h2>Overview</h2>
<img src="{args.png}" alt="Tearsheet v2" />
{missing_note}
{"<h2>Top Sector Exposures (last)</h2>" + top_sector_html if top_sector_html else ""}
</body></html>"""
    (outdir / args.html).write_text(html, encoding="utf-8")
    print(f"[OK] Wrote: {outdir / args.png}")
    print(f"[OK] Wrote: {outdir / args.html}")


if __name__ == "__main__":
    main()

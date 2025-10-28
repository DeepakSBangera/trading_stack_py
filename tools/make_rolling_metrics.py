from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

# ✅ Centralized date utils
from tradingstack.utils.dates import coerce_date_index


# ---------- IO helpers ----------
def _load_portfolio(p: Path) -> tuple[pd.Series, pd.Series | None]:
    """
    Return (nav_series, returns_series or None). Index is daily tz-naive date.
    - Accepts NAV columns: nav_net, nav_gross, _nav, nav
    - If only returns are available, synth NAV is built from returns
    """
    df = pd.read_parquet(p)
    df = coerce_date_index(df, date_col="date")  # normalizes index to daily tz-naive

    nav_col = next(
        (c for c in ("nav_net", "nav_gross", "_nav", "nav") if c in df.columns), None
    )
    if (
        nav_col is None
        and "ret_net" not in df.columns
        and "ret_gross" not in df.columns
    ):
        raise ValueError("portfolio parquet must contain nav_* column or ret_* column")

    if "ret_net" in df.columns:
        rets = pd.to_numeric(df["ret_net"], errors="coerce")
    elif "ret_gross" in df.columns:
        rets = pd.to_numeric(df["ret_gross"], errors="coerce")
    else:
        rets = None

    if nav_col is None:
        nav = (1.0 + (rets.fillna(0.0) if rets is not None else 0.0)).cumprod()
    else:
        nav = pd.to_numeric(df[nav_col], errors="coerce")

    return nav, rets


# ---------- local rolling implementation (fallback) ----------
def _rolling_vol(rets: pd.Series, win: int) -> pd.Series:
    return rets.rolling(win).std(ddof=0) * math.sqrt(252.0)


def _rolling_sharpe(rets: pd.Series, win: int) -> pd.Series:
    mu = rets.rolling(win).mean()
    sd = rets.rolling(win).std(ddof=0)
    return (mu / sd) * math.sqrt(252.0)


def _rolling_sortino(rets: pd.Series, win: int) -> pd.Series:
    mu = rets.rolling(win).mean()
    dn = rets.copy()
    dn[dn > 0] = 0.0
    dsd = dn.rolling(win).std(ddof=0).replace(0.0, np.nan)
    return (mu / dsd) * math.sqrt(252.0)


def _rolling_mdd_from_nav(nav: pd.Series, win: int) -> pd.Series:
    def mdd_window(x: np.ndarray) -> float:
        arr = pd.Series(x)
        peak = arr.cummax()
        dd = (arr / peak) - 1.0
        return float(dd.min())

    return nav.rolling(win).apply(mdd_window, raw=True)


def _local_compute(nav: pd.Series, rets: pd.Series | None, win: int) -> pd.DataFrame:
    if rets is None:
        rets = nav.pct_change()
    rets = pd.to_numeric(rets, errors="coerce")

    out = pd.DataFrame(index=nav.index)
    out["rolling_vol"] = _rolling_vol(rets, win)
    out["rolling_sharpe"] = _rolling_sharpe(rets, win)
    out["rolling_sortino"] = _rolling_sortino(rets, win)
    out["rolling_mdd"] = _rolling_mdd_from_nav(nav, win)
    rs = out["rolling_sharpe"]
    out["regime"] = rs.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    return out


def main():
    ap = argparse.ArgumentParser(description="Build rolling metrics parquet + summary")
    ap.add_argument("--portfolio", default="reports/portfolio_v2.parquet")
    ap.add_argument("--outdir", default="reports")
    ap.add_argument("--window", type=int, default=252)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_parquet = outdir / "rolling_metrics.parquet"
    out_summary = outdir / "rolling_metrics_summary.txt"

    nav, rets = _load_portfolio(Path(args.portfolio))

    # Prefer project module if it exists; otherwise local implementation
    try:
        from tradingstack.metrics.rolling import (
            compute_rolling_metrics_from_nav,  # type: ignore
        )

        try:
            df_roll = compute_rolling_metrics_from_nav(nav, window=args.window)
        except TypeError:
            df_roll = compute_rolling_metrics_from_nav(nav, win=args.window)
    except Exception:
        df_roll = _local_compute(nav, rets, args.window)

    df_roll = df_roll.sort_index()
    df_roll.to_parquet(out_parquet)

    # Summary
    last_dt = df_roll.index.max()
    lines = []
    lines.append(f"Rows: {len(df_roll)}  Start: {df_roll.index.min()}  End: {last_dt}")

    def _last(col: str) -> float | None:
        if col not in df_roll.columns or not df_roll[col].notna().any():
            return None
        return float(df_roll[col].dropna().iloc[-1])

    sr = _last("rolling_sharpe")
    so = _last("rolling_sortino")
    rv = _last("rolling_vol")
    mdd = _last("rolling_mdd")
    rg = _last("regime")

    def _fmt(x, pct=False):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "NA"
        return f"{x:.2%}" if pct else f"{x:.2f}"

    lines.append(
        f"Last ({str(last_dt)[:10]}): Sharpe={_fmt(sr)}, Sortino={_fmt(so)}, "
        f"Vol={_fmt(rv)}, MDD={_fmt(mdd, pct=True)}, Regime={int(rg) if rg is not None and not np.isnan(rg) else 'NA'}"
    )
    lines.append("\nNaN counts:")
    lines.append(str(df_roll.isna().sum()))

    out_summary.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Wrote: {out_parquet}")
    print(f"[OK] Wrote: {out_summary}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

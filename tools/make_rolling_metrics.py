from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from tradingstack.metrics.rolling import compute_rolling_metrics_from_nav


def _normalize_date_index(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Accept tz-aware UTC or tz-naive 'date', set tz-naive DatetimeIndex named 'date'."""
    if date_col in df.columns:
        dt = pd.to_datetime(df[date_col], utc=True, errors="coerce")
        # strip tz → tz-naive
        try:
            dt = dt.dt.tz_convert(None)
        except Exception:
            pass
        try:
            dt = dt.dt.tz_localize(None)
        except Exception:
            pass
        df = df.assign(**{date_col: dt}).dropna(subset=[date_col]).sort_values(date_col)
        df = df.drop_duplicates(subset=[date_col]).set_index(date_col)
    else:
        idx = pd.to_datetime(df.index, utc=True, errors="coerce")
        try:
            idx = idx.tz_convert(None)
        except Exception:
            pass
        try:
            idx = idx.tz_localize(None)
        except Exception:
            pass
        df.index = idx
        df = df.dropna(axis=0, how="any").sort_index()
    df.index.name = "date"
    return df


DEFAULT_CFG = Path("config/rolling.json")


def load_cfg(cfg_path: Path) -> dict:
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _print_summary(df: pd.DataFrame) -> str:
    lines = []
    lines.append(f"Rows: {len(df)}  Start: {df.index.min().date()}  End: {df.index.max().date()}")
    last = df.dropna().tail(1)
    if not last.empty:
        dt = last.index[-1].date()
        v = last.iloc[-1]
        lines.append(
            f"Last ({dt}): Sharpe={v['rolling_sharpe']:.2f}, "
            f"Sortino={v['rolling_sortino']:.2f}, Vol={v['rolling_vol']:.2%}, "
            f"MDD={v['rolling_mdd']:.2%}, Regime={int(v['regime'])}"
        )
    lines.append("")
    lines.append("NaN counts:")
    lines.append(str(df.isna().sum()))
    return "\n".join(lines)


def main(cfg_file: str | None = None):
    cfg_path = Path(cfg_file) if cfg_file else DEFAULT_CFG
    cfg = load_cfg(cfg_path)

    nav_path = Path(cfg["nav_file"])
    if not nav_path.exists():
        print(f"[ERROR] NAV parquet not found: {nav_path}", file=sys.stderr)
        sys.exit(2)

    df = pd.read_parquet(nav_path)
    df = _normalize_date_index(df, "date")

    requested = cfg.get("nav_col", "nav_net")
    nav_col = None
    candidates = [requested, "nav_net", "nav_gross", "_nav"]
    for c in candidates:
        if c in df.columns:
            nav_col = c
            break
    if nav_col is None:
        print(
            f"[ERROR] None of the NAV columns found {candidates} in {nav_path.name}",
            file=sys.stderr,
        )
        sys.exit(3)
    if nav_col != requested:
        print(
            f"[WARN] Requested nav_col '{requested}' not found. Using '{nav_col}' instead.",
            file=sys.stderr,
        )

    out = compute_rolling_metrics_from_nav(
        df=df,
        nav_col=nav_col,
        ret_window_sharpe=int(cfg.get("ret_window_sharpe", 252)),
        ret_window_sortino=int(cfg.get("ret_window_sortino", 252)),
        vol_window=int(cfg.get("vol_window", 63)),
        dd_window=int(cfg.get("dd_window", 252)),
        annualization=int(cfg.get("annualization", 252)),
        rf_per_period=float(cfg.get("rf_per_period", 0.0)),
        regime_method=str(cfg.get("regime_method", "sma_crossover")),
        regime_fast=int(cfg.get("regime_fast", 50)),
        regime_slow=int(cfg.get("regime_slow", 200)),
    )

    out_path = Path(cfg.get("out_parquet", "reports/rolling_metrics.parquet"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=True)

    summary = _print_summary(out)
    sum_path = Path(cfg.get("out_summary", "reports/rolling_metrics_summary.txt"))
    sum_path.write_text(summary, encoding="utf-8")

    print(f"[OK] Wrote: {out_path}")
    print(f"[OK] Wrote: {sum_path}")
    print()
    print(summary)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)

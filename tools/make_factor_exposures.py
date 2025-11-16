from __future__ import annotations

import pathlib

# tools/make_factor_exposures.py
# Build factor panels and write reports/*
# --- repo-path bootstrap (top of file) ---
import sys

_repo_root = pathlib.Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
# -----------------------------------------

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from tradingstack.factors import (
    load_sector_mapping,
    rolling_sector_exposures_from_weights,
)
from tradingstack.factors import (
    momentum_12_1_proxy as momentum_proxy_12_1_from_nav,
)
from tradingstack.factors import (
    quality_inverse_downside_vol as quality_proxy_inv_downside_vol,
)


def _find_latest(glob_pattern: str) -> Path | None:
    paths = list(Path(".").glob(glob_pattern))
    return max(paths, key=lambda p: p.stat().st_mtime) if paths else None


def _ensure_utc_index(df: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    if isinstance(df, pd.Series):
        idx = df.index
        if not isinstance(idx, pd.DatetimeIndex):
            raise ValueError("Expected a DatetimeIndex on the Series.")
        df.index = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
        return df
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise ValueError("Expected a DatetimeIndex on the DataFrame.")
    df.index = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    return df


def _dedupe_index_last(obj: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    if isinstance(obj, pd.Series):
        obj = obj[~obj.index.duplicated(keep="last")]
        return obj.sort_index()
    obj = obj[~obj.index.duplicated(keep="last")]
    return obj.sort_index()


def _coerce_date_index(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    if date_col.lower() in cols:
        real = cols[date_col.lower()]
        d = pd.to_datetime(df[real], utc=True, errors="coerce")
        df = df.drop(columns=[real])
        df.index = d
        df = df[~df.index.isna()]
    df.index = pd.to_datetime(df.index, utc=True)
    return df.sort_index()


def _read_csv_with_date_index(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    return _coerce_date_index(df, "date")


def _get_nav_series(port_df: pd.DataFrame) -> pd.Series:
    candidates = ["nav", "equity", "portfolio", "portfolio_nav", "value"]
    lowmap = {c.lower(): c for c in port_df.columns}
    for k in candidates:
        if k in lowmap:
            s = pd.to_numeric(port_df[lowmap[k]], errors="coerce")
            if s.notna().any():
                out = pd.Series(s.values, index=port_df.index)
                return _dedupe_index_last(_ensure_utc_index(out))
    for c in port_df.columns:
        s = pd.to_numeric(port_df[c], errors="coerce")
        if s.notna().any():
            out = pd.Series(s.values, index=port_df.index)
            return _dedupe_index_last(_ensure_utc_index(out))
    raise ValueError("Could not infer a NAV/equity column from portfolio CSV.")


def _get_weights_table(wdf: pd.DataFrame) -> pd.DataFrame:
    wdf = _coerce_date_index(wdf.copy(), "date")
    for c in wdf.columns:
        wdf[c] = pd.to_numeric(wdf[c], errors="coerce")
    wdf = wdf.fillna(0.0)
    wdf = _ensure_utc_index(wdf)
    return _dedupe_index_last(wdf)


def _pct_change_robust(nav: pd.Series) -> pd.Series:
    rets = nav.pct_change(fill_method=None)
    rets = rets.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    rets.index = pd.to_datetime(rets.index, utc=True)
    return _dedupe_index_last(rets)


def _scalar(x, default=float("nan")) -> float:
    try:
        if isinstance(x, pd.Series):
            x = pd.to_numeric(x, errors="coerce").dropna()
            if x.empty:
                return default
            return float(x.iloc[-1])
        return float(x)
    except Exception:
        return default


def _safe_write_csv(
    df: pd.DataFrame, path: Path, retries: int = 5, sleep_s: float = 0.6
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    for _k in range(retries):
        try:
            df.to_csv(tmp, index=True)
            os.replace(tmp, path)
            return path
        except PermissionError:
            try:
                if tmp.exists():
                    tmp.unlink(missing_ok=True)
            except Exception:
                pass
            time.sleep(sleep_s)
    stamped = path.with_name(f"{path.stem}_{int(time.time())}{path.suffix}")
    df.to_csv(stamped, index=True)
    return stamped


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build factor exposures panels.")
    p.add_argument("--portfolio-csv", type=str, default="")
    p.add_argument("--weights-csv", type=str, default="")
    p.add_argument("--mapping-csv", type=str, default="")
    p.add_argument("--window", type=int, default=63, help="Rolling window (days).")
    p.add_argument("--outdir", type=str, default="reports", help="Output directory")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    port_csv = (
        Path(args.portfolio_csv)
        if args.portfolio_csv
        else _find_latest("reports/portfolioV2_*.csv")
    )
    weights_csv = (
        Path(args.weights_csv)
        if args.weights_csv
        else _find_latest("reports/portfolioV2_*_weights.csv")
    )
    mapping_csv = (
        Path(args.mapping_csv)
        if args.mapping_csv
        else Path("config/sector_mapping.csv")
    )

    if port_csv is None or not port_csv.exists():
        raise FileNotFoundError(
            "Portfolio CSV not found. Provide --portfolio-csv or run Report-PortfolioV2 first."
        )
    if weights_csv is None or not weights_csv.exists():
        raise FileNotFoundError(
            "Weights CSV not found. Provide --weights-csv or run Report-PortfolioV2 first."
        )
    if not mapping_csv.exists():
        raise FileNotFoundError(f"Sector mapping CSV not found at '{mapping_csv}'.")

    port = _read_csv_with_date_index(port_csv)
    port = _ensure_utc_index(_dedupe_index_last(port))
    weights = _get_weights_table(pd.read_csv(weights_csv))

    common_idx = port.index.intersection(weights.index)
    if common_idx.empty:
        raise SystemExit("No overlapping dates between portfolio and weights.")
    port = _dedupe_index_last(_ensure_utc_index(port.loc[common_idx]))
    weights = _dedupe_index_last(_ensure_utc_index(weights.loc[common_idx]))

    mapping: dict[str, str] = load_sector_mapping(mapping_csv)
    sector_roll = rolling_sector_exposures_from_weights(
        weights, mapping, window=args.window
    )
    if (
        isinstance(sector_roll.index, pd.DatetimeIndex)
        and sector_roll.index.tz is not None
    ):
        sector_roll.index = sector_roll.index.tz_localize(None)
    nav = _get_nav_series(port)
    nav = pd.Series(pd.to_numeric(nav, errors="coerce").values, index=port.index)
    nav = _dedupe_index_last(_ensure_utc_index(nav))
    mom = momentum_proxy_12_1_from_nav(nav)
    mom.name = "mom_12_1_proxy"
    rets = _pct_change_robust(nav)
    qual = quality_proxy_inv_downside_vol(rets, window=args.window)
    qual.name = "quality_inv_downside_vol"

    base_idx = (
        pd.DatetimeIndex(
            sector_roll.index.tz_localize(
                "UTC", nonexistent="shift_forward", ambiguous="NaT"
            )
        )
        .union(mom.index)
        .union(qual.index)
    )
    base_idx = pd.DatetimeIndex(base_idx.unique()).sort_values()

    out = pd.DataFrame(index=base_idx)
    out = out.join(sector_roll, how="left")
    out["mom_12_1_proxy"] = mom.reindex(out.index)
    out["quality_inv_downside_vol"] = qual.reindex(out.index)
    out = _dedupe_index_last(out)
    out = out.loc[:, ~out.columns.duplicated(keep="last")]

    out_parquet = outdir / "factor_exposures.parquet"
    out_csv = outdir / "factor_exposures.csv"
    out_txt = outdir / "factor_exposures_summary.txt"

    out.to_parquet(out_parquet)
    written_csv = _safe_write_csv(out.round(6), out_csv)

    last_row = out.iloc[-1]
    mom_last = _scalar(last_row.get("mom_12_1_proxy"))
    qlt_last = _scalar(last_row.get("quality_inv_downside_vol"))
    sector_cols = [
        c
        for c in out.columns
        if c.lower() not in ("mom_12_1_proxy", "quality_inv_downside_vol")
    ]
    sector_sum_last = (
        _scalar(pd.to_numeric(last_row[sector_cols], errors="coerce").sum())
        if sector_cols
        else float("nan")
    )

    with open(out_txt, "w", encoding="utf-8") as fh:
        fh.write(
            "\n".join(
                [
                    f"Rows: {len(out):d}",
                    f"Date range: {out.index.min().date()} \u2192 {out.index.max().date()}",
                    f"  Momentum (12-1) proxy: {mom_last:.4f}",
                    f"  Quality (inv downside vol): {qlt_last:.4f}",
                    f"  Sector weights (last row sum): {sector_sum_last:.4f}",
                ]
            )
        )

    print(f"[OK] Wrote: {out_parquet}")
    print(f"[OK] Wrote CSV: {written_csv}")
    print(f"[OK] Wrote: {out_txt}")


if __name__ == "__main__":
    main()

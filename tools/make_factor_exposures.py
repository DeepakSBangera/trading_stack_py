from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tradingstack.factors import (
    load_sector_mapping,
    momentum_proxy_12_1_from_nav,
    quality_proxy_inv_downside_vol,
    rolling_sector_exposures_from_weights,
)


def _read_weights(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if {"date", "ticker", "weight"}.issubset(df.columns):
        return df[["date", "ticker", "weight"]].copy()

    if "date" not in df.columns and isinstance(df.index, (pd.DatetimeIndex, pd.Index)):
        wide = df.reset_index().rename(columns={"index": "date"})
    else:
        wide = df.copy()

    if "date" not in wide.columns:
        raise ValueError("weights parquet must have 'date' column or DatetimeIndex")

    date_col = "date"
    tickers = [c for c in wide.columns if c != date_col]
    long = wide.melt(
        id_vars=[date_col], value_vars=tickers, var_name="ticker", value_name="weight"
    )
    return long


def _read_nav(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    cols = {c.strip().lower(): c for c in df.columns}
    date_col = cols.get("date", "date")
    nav_col = None
    for cand in ("nav_net", "nav_gross", "_nav", "nav"):
        if cand in df.columns:
            nav_col = cand
            break
    if nav_col is None:
        raise ValueError(
            "portfolio parquet missing nav column (nav_net/nav_gross/_nav/nav)"
        )

    sdate = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    try:
        sdate = sdate.dt.tz_convert(None)
    except Exception:
        pass
    try:
        sdate = sdate.dt.tz_localize(None)
    except Exception:
        pass
    df = df.copy()
    df[date_col] = pd.to_datetime(sdate.dt.date)
    return df.rename(columns={date_col: "date"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="reports/weights_v2_norm.parquet")
    ap.add_argument("--portfolio", default="reports/portfolio_v2.parquet")
    ap.add_argument("--sector-mapping", default="config/sector_mapping.csv")
    ap.add_argument("--window", type=int, default=63)
    ap.add_argument("--skip-days", type=int, default=21)
    ap.add_argument("--trading-days-12m", type=int, default=252)
    ap.add_argument("--outdir", default="reports")
    args = ap.parse_args()

    weights_p = Path(args.weights)
    portfolio_p = Path(args.portfolio)
    mapping_p = Path(args.sector_mapping)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not weights_p.exists():
        raise SystemExit(f"Missing weights parquet: {weights_p}")
    if not portfolio_p.exists():
        raise SystemExit(f"Missing portfolio parquet: {portfolio_p}")

    weights = _read_weights(weights_p)
    port = _read_nav(portfolio_p)

    if not mapping_p.exists():
        tickers = sorted(
            {str(t) for t in weights["ticker"].dropna().astype(str).unique().tolist()}
        )
        mapping_df = pd.DataFrame(
            {"ticker": tickers, "sector": ["Unknown"] * len(tickers)}
        )
        mapping_p.parent.mkdir(parents=True, exist_ok=True)
        mapping_df.to_csv(mapping_p, index=False)
        raise SystemExit(
            f"Sector mapping not found; wrote template at {mapping_p}. Fill 'sector' and re-run."
        )

    mapping = load_sector_mapping(mapping_p)

    sector_roll = rolling_sector_exposures_from_weights(
        weights, mapping, window=args.window
    )

    rcol = None
    for cand in ("ret_net", "ret_gross"):
        if cand in port.columns:
            rcol = cand
            break
    if rcol is None:
        port = port.sort_values("date")
        if "nav_net" in port.columns:
            port["ret_net"] = port["nav_net"].pct_change()
            rcol = "ret_net"
        elif "nav_gross" in port.columns:
            port["ret_gross"] = port["nav_gross"].pct_change()
            rcol = "ret_gross"
        else:
            raise SystemExit("Could not derive returns from NAV.")

    port = port.set_index("date").sort_index()
    nav_col = (
        "nav_net"
        if "nav_net" in port.columns
        else ("nav_gross" if "nav_gross" in port.columns else None)
    )
    if nav_col is None:
        raise SystemExit(
            "Could not find nav column (nav_net/nav_gross) in portfolio parquet"
        )

    mom = momentum_proxy_12_1_from_nav(
        port[nav_col], trading_days_12m=args.trading_days_12m, skip_days=args.skip_days
    )
    qual = quality_proxy_inv_downside_vol(port[rcol], window=args.window)

    out = port[[rcol]].copy()
    out = out.join(sector_roll, how="left")
    out["mom_12_1_proxy"] = mom
    out["quality_inv_downside_vol"] = qual

    out_path = outdir / "factor_exposures.parquet"
    out.to_parquet(out_path)

    last_date = out.index.max()
    last = out.loc[last_date]
    top_secs = sorted(
        (
            (c, float(last[c]))
            for c in out.columns
            if c.startswith("sector_") and pd.notnull(last[c])
        ),
        key=lambda x: abs(x[1]),
        reverse=True,
    )[:5]

    summary_lines = [
        f"Rows: {len(out):,}   Start: {out.index.min().date()}   End: {out.index.max().date()}",
        "",
        f"Last date: {last_date.date()}",
        (
            f"  Momentum (12-1) proxy: {float(last['mom_12_1_proxy']):.4f}"
            if pd.notnull(last["mom_12_1_proxy"])
            else "  Momentum (12-1) proxy: NaN"
        ),
        (
            f"  Quality (inv downside vol): {float(last['quality_inv_downside_vol']):.4f}"
            if pd.notnull(last["quality_inv_downside_vol"])
            else "  Quality (inv downside vol): NaN"
        ),
        "",
        "Top sector exposures (rolling avg weights):",
    ]
    for name, val in top_secs:
        summary_lines.append(f"  - {name}: {val:.4f}")

    (outdir / "factor_exposures_summary.txt").write_text(
        "\n".join(summary_lines), encoding="utf-8"
    )

    print(f"[OK] Wrote: {out_path}")
    print(f"[OK] Wrote: {outdir/'factor_exposures_summary.txt'}")


if __name__ == "__main__":
    main()

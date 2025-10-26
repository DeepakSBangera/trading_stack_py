# tools/report_portfolio_synth_v2.py
# Drop-in replacement (v2.2)
# - Robust ticker file lookup (flat or nested)
# - Robust datetime index (UTC) handling
# - Full-history momentum, slice outputs to --start
# - Clean turnover masking (no .assign), no tz warnings

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------
# Utilities: robust I/O & dates
# ----------------------------


def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure df is indexed by a UTC DatetimeIndex named 'date', sorted ascending."""
    if "date" in df.columns:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
        df = df.dropna(subset=["date"]).set_index("date")
    elif isinstance(df.index, pd.DatetimeIndex):
        idx = df.index
        if idx.tz is None:
            df.index = idx.tz_localize("UTC")
        else:
            df.index = idx.tz_convert("UTC")
    else:
        for col in ("timestamp", "time", "datetime"):
            if col in df.columns:
                df = df.copy()
                df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
                df = df.dropna(subset=[col]).set_index(col)
                break
        else:
            raise TypeError("Prices require a datetime column ('date') or a DatetimeIndex.")
    df = df.sort_index()
    df.index.name = "date"
    return df


def find_price_path(ticker: str, prices_root: str) -> str:
    """Map 'AXISBANK.NS' -> 'AXISBANK_NS.parquet' and locate under prices_root (flat first, then recursive)."""
    base = ticker.replace(".", "_")
    flat = os.path.join(prices_root, f"{base}.parquet")
    if os.path.exists(flat):
        return flat
    hits = glob.glob(os.path.join(prices_root, "**", f"{base}.parquet"), recursive=True)
    if hits:
        hits.sort(key=len)  # prefer top-most match
        return hits[0]
    raise FileNotFoundError(f"price parquet missing for {ticker}: {flat}")


def load_prices_for(ticker: str, prices_root: str) -> pd.DataFrame:
    """Load a single symbol's price parquet. Returns DF with UTC DatetimeIndex and column 'close' (float)."""
    path = find_price_path(ticker, prices_root)
    df = pd.read_parquet(path)
    df = ensure_datetime_index(df)

    cols_lower = {c.lower(): c for c in df.columns}
    close_col = cols_lower.get("close")
    if close_col is None:
        for candidate in ("adj_close", "adjusted_close", "close_price", "last"):
            if candidate in cols_lower:
                close_col = cols_lower[candidate]
                break
    if close_col is None:
        raise KeyError(f"'close' column not found in {path}. Columns: {list(df.columns)}")

    out = df[[close_col]].rename(columns={close_col: "close"}).dropna()
    out["close"] = out["close"].astype(float)
    return out


# ---------- Rebalance calendars (no tz warnings) ----------


def month_end_dates(ix: pd.DatetimeIndex) -> pd.DatetimeIndex:
    # last available date per (year, month)
    ix = ix.tz_convert("UTC")
    s = pd.Series(0, index=ix)
    me = s.groupby([ix.year, ix.month]).apply(lambda x: x.index.max())
    return pd.DatetimeIndex(me.to_list(), tz="UTC")


def quarter_end_dates(ix: pd.DatetimeIndex) -> pd.DatetimeIndex:
    ix = ix.tz_convert("UTC")
    s = pd.Series(0, index=ix)
    qe = s.groupby([ix.year, ix.quarter]).apply(lambda x: x.index.max())
    return pd.DatetimeIndex(qe.to_list(), tz="UTC")


def week_end_dates(ix: pd.DatetimeIndex) -> pd.DatetimeIndex:
    # last available date per ISO (year, week)
    ix = ix.tz_convert("UTC")
    iso = ix.isocalendar()
    s = pd.Series(0, index=ix)
    we = s.groupby([iso["year"].to_numpy(), iso["week"].to_numpy()]).apply(lambda x: x.index.max())
    return pd.DatetimeIndex(we.to_list(), tz="UTC")


def make_rebalance_calendar(ix: pd.DatetimeIndex, mode: str) -> pd.DatetimeIndex:
    m = mode.upper()
    if m == "ME":
        c = month_end_dates(ix)
    elif m == "WE":
        c = week_end_dates(ix)
    elif m == "QE":
        c = quarter_end_dates(ix)
    else:
        raise ValueError(f"Unsupported rebalance mode: {mode} (use ME|WE|QE)")
    return c.intersection(ix)


# ----------------------------
# Portfolio construction (Top-N momentum)
# ----------------------------


def compute_momentum(px: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """close/close.shift(lookback) - 1"""
    return px / px.shift(lookback) - 1.0


def build_topn_weights(
    px_wide: pd.DataFrame, rebalance_dates: pd.DatetimeIndex, lookback: int, topn: int
) -> pd.DataFrame:
    """Equal-weight Top-N momentum names at each rebalance; carry forward until next."""
    mom = compute_momentum(px_wide, lookback)
    weights = pd.DataFrame(0.0, index=px_wide.index, columns=px_wide.columns, dtype=float)
    last_w = pd.Series(0.0, index=px_wide.columns, dtype=float)

    for d in rebalance_dates:
        if d not in mom.index:
            continue
        mrow = mom.loc[d].dropna()
        if mrow.empty:
            continue
        top = mrow.sort_values(ascending=False).head(topn).index
        w = pd.Series(0.0, index=px_wide.columns, dtype=float)
        if len(top) > 0:
            w.loc[top] = 1.0 / float(len(top))
        last_w = w
        weights.loc[d:] = last_w.values

    return weights.ffill().fillna(0.0)


def apply_transaction_costs(
    weights: pd.DataFrame, rets: pd.DataFrame, rebalance_dates: pd.DatetimeIndex, cost_bps: float
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Gross ret = w_{t-1}·r_t ; net ret = gross - turnover*cost_bps/10000 on rebalance days; turnover = |Δw|/2."""
    w_shift = weights.shift(1).fillna(0.0)
    gross = (w_shift * rets).sum(axis=1)

    delta = weights.diff().abs().sum(axis=1)
    turnover = delta / 2.0

    # Only charge turnover on rebalance days
    mask = pd.Series(False, index=weights.index)
    mask.loc[rebalance_dates] = True
    turnover = turnover.where(mask, 0.0)

    costs = turnover * (cost_bps / 10000.0)
    net = gross - costs
    return gross, net, turnover


# ----------------------------
# Main
# ----------------------------


def read_universe(csv_path: str) -> list[str]:
    df = pd.read_csv(csv_path)
    cols = {c.strip().lower(): c for c in df.columns}
    tcol = cols.get("ticker")
    if tcol is None:
        raise ValueError(f"Universe CSV must have a 'ticker' header: {csv_path}")
    syms = df[tcol].dropna().astype(str).str.strip()
    syms = [s for s in syms if s]
    if not syms:
        raise ValueError(f"No tickers found in {csv_path}")
    return pd.Index(syms).unique().tolist()


def assemble_panel(tickers: list[str], prices_root: str) -> pd.DataFrame:
    """Return wide price panel: index=date (UTC), columns=tickers, values=close."""
    series = []
    names = []
    for t in tickers:
        df = load_prices_for(t, prices_root)
        s = df["close"].copy()
        series.append(s)
        names.append(t)
    panel = pd.concat(series, axis=1)
    panel.columns = names
    panel = panel.sort_index()
    panel = panel.dropna(how="all")
    return panel


def main():
    p = argparse.ArgumentParser(
        description="Top-N momentum portfolio (synthetic V2, pre-history aware)"
    )
    p.add_argument("--universe-csv", required=True, help="CSV with header 'ticker'")
    p.add_argument(
        "--prices-root", required=True, help="Root folder of price parquet files (can be nested)"
    )
    p.add_argument("--start", required=True, help="Backtest start date (YYYY-MM-DD)")
    p.add_argument(
        "--lookback", type=int, default=126, help="Momentum lookback in trading days (default: 126)"
    )
    p.add_argument("--top-n", type=int, default=4, help="Number of names to hold (default: 4)")
    p.add_argument(
        "--rebalance",
        choices=["ME", "WE", "QE"],
        default="ME",
        help="Rebalance schedule: ME/WE/QE (default: ME)",
    )
    p.add_argument(
        "--cost-bps",
        type=float,
        default=10.0,
        help="One-way transaction cost in bps, applied on turnover at rebalances (default: 10)",
    )
    p.add_argument("--outdir", required=True, help="Directory to write outputs")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load full history first (so lookback works), then slice outputs later
    tickers = read_universe(args.universe_csv)
    panel_full = assemble_panel(tickers, args.prices_root)
    panel_full.index = pd.to_datetime(panel_full.index, utc=True)

    # Rebalance calendar from full index; keep only on/after start
    start = pd.to_datetime(args.start, utc=True, errors="coerce")
    if pd.isna(start):
        raise ValueError(f"Invalid --start date: {args.start}")
    rcal_all = make_rebalance_calendar(panel_full.index, args.rebalance)
    rcal = rcal_all[rcal_all >= start]
    if len(rcal) == 0:
        raise ValueError(
            "No rebalance dates on/after the requested start date. Try an earlier start or ensure data exists."
        )

    # Returns from full history; build weights using rebalances >= start
    rets_full = panel_full.pct_change().fillna(0.0)
    weights_full = build_topn_weights(panel_full, rcal, lookback=args.lookback, topn=args.top_n)

    # Apply costs using full returns (weights constant outside rebalances)
    gross_full, net_full, turnover_full = apply_transaction_costs(
        weights_full, rets_full, rcal, cost_bps=args.cost_bps
    )

    # Slice outputs to start date (post-computation)
    mask = gross_full.index >= start
    gross = gross_full[mask]
    net = net_full[mask]
    turnover = turnover_full[mask]
    weights = weights_full.loc[weights_full.index >= start]

    if gross.empty:
        raise ValueError("No price data on/after the requested start date.")

    # NAV series
    nav_gross = (1.0 + gross).cumprod()
    nav_net = (1.0 + net).cumprod()

    # Write outputs
    portfolio_df = pd.DataFrame(
        {
            "nav_gross": nav_gross,
            "nav_net": nav_net,
            "ret_gross": gross,
            "ret_net": net,
            "turnover": turnover,
        }
    )
    portfolio_df.index.name = "date"
    portfolio_df.to_parquet(outdir / "portfolio_v2.parquet")

    weights.index.name = "date"
    weights.to_parquet(outdir / "weights_v2.parquet")

    # Trades on rebalance days (before/after/turnover)
    trades_rows: list[dict] = []
    weights_prev = weights.shift(1).fillna(0.0)
    delta = weights - weights_prev
    for d in rcal:
        if d not in weights.index:
            continue
        row_before = weights_prev.loc[d]
        row_after = weights.loc[d]
        dlt = (row_after - row_before).rename("delta")
        changed = dlt[~np.isclose(dlt.values, 0.0)]
        if len(changed) == 0:
            continue
        one_way_turnover = np.abs(dlt).sum() / 2.0
        for tkr, ch in changed.items():
            trades_rows.append(
                {
                    "date": d,
                    "ticker": tkr,
                    "w_before": float(row_before.get(tkr, 0.0)),
                    "w_after": float(row_after.get(tkr, 0.0)),
                    "delta": float(ch),
                    "turnover_day": float(one_way_turnover),
                    "cost_bps_applied": float(args.cost_bps),
                }
            )
    if trades_rows:
        trades_df = pd.DataFrame(trades_rows).sort_values(["date", "ticker"])
        trades_df.to_parquet(outdir / "trades_v2.parquet")

    last = portfolio_df.iloc[-1]
    print(
        f"Done. {len(portfolio_df):,} days | "
        f"NAV_net={last['nav_net']:.4f} | last_ret_net={last['ret_net']:.5f} | "
        f"rebalances(on/after start)={len(rcal)} | outputs → {outdir}"
    )


if __name__ == "__main__":
    main()

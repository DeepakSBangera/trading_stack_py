# tools/w6_portfolio_compare.py
# Compares Equal-Weight (EW) vs Ledoit–Wolf Shrinkage Minimum-Variance (LW-GMV)
# - Uses full price history for estimation windows, then slices outputs to --start
# - Long-only projection (clip negatives to 0 then renormalize)
# - Rebalance schedule: ME | WE | QE
# - Costs applied on turnover at rebalances (bps)
# Outputs (under --outdir):
#   wk6_portfolio_compare.parquet  (EW/LW NAV, returns, turnover)
#   wk6_weights_ew.parquet         (wide)
#   wk6_weights_lw.parquet         (wide)

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

# ---------- basics reused ----------


def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if "date" in df.columns:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
        df = df.dropna(subset=["date"]).set_index("date")
    elif isinstance(df.index, pd.DatetimeIndex):
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
    else:
        for col in ("timestamp", "time", "datetime"):
            if col in df.columns:
                df = df.copy()
                df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
                df = df.dropna(subset=[col]).set_index(col)
                break
        else:
            raise TypeError("Need a datetime column or DatetimeIndex")
    df = df.sort_index()
    df.index.name = "date"
    return df


def find_price_path(ticker: str, prices_root: str) -> str:
    base = ticker.replace(".", "_")
    flat = os.path.join(prices_root, f"{base}.parquet")
    if os.path.exists(flat):
        return flat
    hits = glob.glob(os.path.join(prices_root, "**", f"{base}.parquet"), recursive=True)
    if hits:
        hits.sort(key=len)
        return hits[0]
    raise FileNotFoundError(f"price parquet missing for {ticker}: {flat}")


def load_prices_for(ticker: str, prices_root: str) -> pd.Series:
    df = pd.read_parquet(find_price_path(ticker, prices_root))
    df = ensure_datetime_index(df)
    cols = {c.lower(): c for c in df.columns}
    c = (
        cols.get("close")
        or cols.get("adj_close")
        or cols.get("adjusted_close")
        or cols.get("close_price")
        or cols.get("last")
    )
    if not c:
        raise KeyError(f"'close' column not found for {ticker}")
    s = df[c].astype(float).rename(ticker).dropna()
    return s


def read_universe(csv_path: str) -> list[str]:
    df = pd.read_csv(csv_path)
    col = {c.lower(): c for c in df.columns}.get("ticker")
    if not col:
        raise ValueError("Universe CSV must have header 'ticker'")
    syms = df[col].dropna().astype(str).str.strip()
    return pd.Index(syms).unique().tolist()


def month_end_dates(ix: pd.DatetimeIndex) -> pd.DatetimeIndex:
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
        raise ValueError("rebalance must be ME|WE|QE")
    return c.intersection(ix)


# ---------- Ledoit–Wolf shrinkage (analytic) ----------


def ledoit_wolf_shrinkage(returns_window: np.ndarray) -> np.ndarray:
    """
    Simple analytic Ledoit–Wolf shrinkage to the identity *scaled by average variance*.
    Inputs:
      returns_window: T x N matrix of demeaned returns
    Returns:
      Sigma_hat (N x N)
    """
    X = returns_window
    T, N = X.shape
    if T < 3:  # guard
        S = np.cov(X, rowvar=False, ddof=1)
        return S
    # sample covariance
    S = np.cov(X, rowvar=False, ddof=1)
    # target: mu*I
    mu = np.trace(S) / N
    F = mu * np.eye(N)

    # phi-hat (variance of the elements of S)
    X2 = X**2
    phi_mat = (X2.T @ X2) / T - S**2
    phi = np.sum(phi_mat)

    # rho-hat
    # off-diagonals
    S_off = S.copy()
    np.fill_diagonal(S_off, 0.0)
    rho = np.sum((S - F) * S_off)

    # gamma-hat
    gamma = np.linalg.norm(S - F, ord="fro") ** 2

    kappa = (phi - rho) / gamma if gamma > 0 else 1.0
    delta = max(0.0, min(1.0, kappa / T))
    Sigma = delta * F + (1.0 - delta) * S
    return Sigma


# ---------- weights, costs, NAV ----------


def long_only_project(w: np.ndarray) -> np.ndarray:
    w = w.copy()
    w[w < 0] = 0.0
    s = w.sum()
    return (w / s) if s > 0 else np.ones_like(w) / len(w)


def gmv_weights_lw(ret_win: pd.DataFrame) -> np.ndarray:
    X = (ret_win - ret_win.mean()).to_numpy()
    Sigma = ledoit_wolf_shrinkage(X)
    # GMV: w ∝ Σ^{-1}·1
    try:
        inv = np.linalg.pinv(Sigma, rcond=1e-8)
    except Exception:
        inv = np.linalg.pinv(Sigma)
    ones = np.ones((Sigma.shape[0], 1))
    raw = (inv @ ones).ravel()
    w = raw / raw.sum() if raw.sum() != 0 else np.ones_like(raw) / len(raw)
    return long_only_project(w)


def ew_weights(n: int) -> np.ndarray:
    return np.ones(n) / n


def apply_costs_and_nav(
    weights_df: pd.DataFrame,
    rets: pd.DataFrame,
    rebal_dates: pd.DatetimeIndex,
    cost_bps: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    w_shift = weights_df.shift(1).fillna(0.0)
    gross = (w_shift * rets).sum(axis=1)
    delta = weights_df.diff().abs().sum(axis=1)
    turnover = (delta / 2.0).where(pd.Series(False, index=rets.index), 0.0)
    mask = pd.Series(False, index=rets.index)
    mask.loc[rebal_dates] = True
    turnover = (delta / 2.0).where(mask, 0.0)
    costs = turnover * (cost_bps / 10000.0)
    net = gross - costs
    return gross, net, turnover


# ---------- panel ----------


def assemble_panel(tickers: list[str], prices_root: str) -> pd.DataFrame:
    ser = [load_prices_for(t, prices_root) for t in tickers]
    panel = pd.concat(ser, axis=1)
    panel = panel.sort_index().dropna(how="all")
    return panel  # index UTC, columns tickers, values close


# ---------- main ----------


def main():
    ap = argparse.ArgumentParser("W6: EW vs LW-GMV portfolio comparison")
    ap.add_argument("--universe-csv", required=True)
    ap.add_argument("--prices-root", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument(
        "--lookback",
        type=int,
        default=252,
        help="estimation window length in trading days",
    )
    ap.add_argument("--rebalance", choices=["ME", "WE", "QE"], default="ME")
    ap.add_argument("--cost-bps", type=float, default=10.0)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    start = pd.to_datetime(args.start, utc=True, errors="coerce")
    if pd.isna(start):
        raise ValueError(f"Bad --start: {args.start}")

    tickers = read_universe(args.universe_csv)
    px = assemble_panel(tickers, args.prices_root)
    px.index = pd.to_datetime(px.index, utc=True)

    # full-history returns
    rets_full = px.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # rebalance calendar from full index; keep those >= start
    rcal_all = make_rebalance_calendar(px.index, args.rebalance)
    rcal = rcal_all[rcal_all >= start]
    if len(rcal) == 0:
        raise ValueError("No rebalance dates on/after start")

    # build weights time series
    W_ew = pd.DataFrame(0.0, index=px.index, columns=tickers, dtype=float)
    W_lw = pd.DataFrame(0.0, index=px.index, columns=tickers, dtype=float)

    for d in rcal:
        # estimation window ends at previous business day
        end_idx = rets_full.index.get_indexer_for([d])
        if len(end_idx) == 1 and end_idx[0] >= 0:
            end_pos = end_idx[0] - 1
        else:
            # fallback: use position just before d
            end_pos = rets_full.index.get_loc(d, method="ffill") - 1
        start_pos = max(0, end_pos - args.lookback + 1)
        win = rets_full.iloc[start_pos : end_pos + 1]
        if win.empty:
            # default to EW if insufficient history
            w_ew = ew_weights(len(tickers))
            w_lw = w_ew
        else:
            w_ew = ew_weights(len(tickers))
            w_lw = gmv_weights_lw(win)

        # apply from d forward
        W_ew.loc[d:] = w_ew
        W_lw.loc[d:] = w_lw

    # slice to start (post-computation)
    W_ew = W_ew.loc[W_ew.index >= start]
    W_lw = W_lw.loc[W_lw.index >= start]
    rets = rets_full.loc[rets_full.index >= start]

    # returns + NAV
    g_ew, n_ew, t_ew = apply_costs_and_nav(W_ew, rets, rcal, args.cost_bps)
    g_lw, n_lw, t_lw = apply_costs_and_nav(W_lw, rets, rcal, args.cost_bps)

    nav_ew = (1.0 + n_ew).cumprod()
    nav_lw = (1.0 + n_lw).cumprod()

    out = pd.DataFrame(
        {
            "ew_nav": nav_ew,
            "ew_ret_net": n_ew,
            "ew_turnover": t_ew,
            "lw_nav": nav_lw,
            "lw_ret_net": n_lw,
            "lw_turnover": t_lw,
        }
    )
    out.index.name = "date"
    out.to_parquet(outdir / "wk6_portfolio_compare.parquet")
    W_ew.to_parquet(outdir / "wk6_weights_ew.parquet")
    W_lw.to_parquet(outdir / "wk6_weights_lw.parquet")

    print(
        f"Done W6 compare. days={len(out):,} | EW_nav={nav_ew.iloc[-1]:.4f} LW_nav={nav_lw.iloc[-1]:.4f} | outputs→{outdir}"
    )


if __name__ == "__main__":
    main()

# tools/report_portfolio_v2.py
# Momentum portfolio backtest with trades log, weights timeline, benchmark,
# position caps/min weights/cash buffer, flexible rebalance, and mgmt fee.

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
import yfinance as yf

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ------------------ arg parsing ------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Momentum backtest (top-N by lookback return, configurable rebalance)."
    )
    p.add_argument(
        "--tickers",
        default="",
        help="Comma-separated tickers, e.g. 'RELIANCE.NS,HDFCBANK.NS,...'",
    )
    p.add_argument(
        "--universe-csv",
        default="",
        help="CSV with column 'ticker' (overrides --tickers if present).",
    )
    p.add_argument(
        "--start", required=True, help="Backtest start date, e.g. 2015-01-01"
    )
    p.add_argument(
        "--lookback",
        type=int,
        default=126,
        help="Lookback days for momentum window (default 126)",
    )
    p.add_argument(
        "--top-n",
        dest="top_n",
        type=int,
        default=4,
        help="Select top-N by lookback return (default 4)",
    )
    p.add_argument(
        "--max-holdings", type=int, default=0, help="Cap total names held (0=off)."
    )
    p.add_argument(
        "--weight-cap",
        type=float,
        default=1.0,
        help="Per-name cap (0–1, default 1.0=no cap).",
    )
    p.add_argument(
        "--min-weight",
        type=float,
        default=0.0,
        help="Drop names below this weight (0–1).",
    )
    p.add_argument(
        "--cash-buffer", type=float, default=0.0, help="Fraction to keep in cash (0–1)."
    )
    p.add_argument(
        "--rebalance",
        choices=["ME", "WE", "QE"],
        default="ME",
        help="ME=month-end, WE=weekly(Fri), QE=quarter-end. Default ME.",
    )
    p.add_argument(
        "--cost-bps",
        dest="cost_bps",
        type=float,
        default=10,
        help="Round-trip cost in bps per rebalance (default 10)",
    )
    p.add_argument(
        "--mgmt-fee-bps",
        dest="mgmt_fee_bps",
        type=float,
        default=0.0,
        help="Annual mgmt fee in bps (charged daily).",
    )
    p.add_argument(
        "--vol-target",
        dest="vol_target",
        type=float,
        default=0.0,
        help="Annualized vol target in %, 0=off",
    )
    p.add_argument(
        "--turnover-band",
        dest="turnover_band",
        type=float,
        default=0.0,
        help="Only change weights if L1 change exceeds this threshold (0–2)",
    )
    p.add_argument(
        "--benchmark",
        default="",
        help="Optional benchmark ticker (e.g., ^NSEI). Empty to disable.",
    )
    p.add_argument(
        "--outdir", default="reports", help="Output directory (default reports)"
    )
    return p.parse_args()


# ------------------ helpers ------------------


def load_universe(args) -> list[str]:
    # CSV overrides --tickers if provided
    if args.universe_csv.strip():
        dfu = pd.read_csv(args.universe_csv)
        if "ticker" not in dfu.columns:
            raise SystemExit("Universe CSV must contain a 'ticker' column.")
        tickers = [str(t).strip() for t in dfu["ticker"].tolist() if str(t).strip()]
        if not tickers:
            raise SystemExit("Universe CSV produced no tickers.")
        return tickers
    # Else use --tickers
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    if not tickers:
        raise SystemExit("No tickers provided (via --tickers or --universe-csv).")
    return tickers


def safe_adj_close(raw, tickers):
    if isinstance(raw.columns, pd.MultiIndex):
        if "Adj Close" not in raw.columns.levels[0]:
            raise SystemExit("No 'Adj Close' in yfinance output.")
        data = raw["Adj Close"].copy()
    else:
        if "Adj Close" not in raw.columns:
            raise SystemExit("No 'Adj Close' in yfinance output (single-index).")
        data = raw["Adj Close"].to_frame()
        if data.shape[1] == 1 and isinstance(tickers, list) and len(tickers) >= 1:
            data.columns = [tickers[0]]
    return data


def cagr_from_equity(eq: pd.Series) -> float:
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    return float(eq.iloc[-1] ** (1.0 / years) - 1.0) if eq.iloc[-1] > 0 else 0.0


def rebalance_index(rets: pd.DataFrame, mode: str) -> pd.DatetimeIndex:
    if mode == "ME":
        idx = rets.resample("ME").last().index
    elif mode == "WE":
        idx = rets.resample("W-FRI").last().index
    else:  # "QE"
        idx = rets.resample("QE").last().index
    return idx.intersection(rets.index)


def cap_min_normalize(
    weights_series: pd.Series,
    max_holdings: int,
    wcap: float,
    wmin: float,
    cash_buffer: float,
) -> pd.Series:
    """
    Apply:
      - limit to max_holdings largest weights (if >0),
      - per-name cap,
      - drop below min weight,
      - renormalize to (1 - cash_buffer), leaving residual as cash.
    """
    w = weights_series.copy()

    # Max holdings
    if max_holdings and max_holdings > 0 and max_holdings < (w > 0).sum():
        # keep the largest positive weights
        keep = w[w > 0].nlargest(max_holdings)
        w[:] = 0.0
        w[keep.index] = keep.values

    # Cap per name
    if wcap < 1.0:
        w = w.clip(upper=wcap)

    # Drop micro weights
    if wmin > 0.0:
        w[w < wmin] = 0.0

    # Renormalize to (1 - cash_buffer)
    s = w.sum()
    target_sum = max(0.0, 1.0 - cash_buffer)
    if s > 0:
        w = w * (target_sum / s)
    else:
        w[:] = 0.0  # all cash

    return w


# ------------------ main ------------------


def main():
    args = parse_args()
    tickers = load_universe(args)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ------- Data -------
    raw = yf.download(tickers, start=args.start, progress=False, auto_adjust=False)
    px = safe_adj_close(raw, tickers).dropna(how="all")
    if px.empty:
        raise SystemExit("No Adjusted Close data returned—check tickers or start date.")

    rets = px.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # ------- Rebalance schedule -------
    rb_dates = rebalance_index(rets, args.rebalance)
    if len(rb_dates) == 0:
        raise SystemExit("No rebalance dates on the selected frequency.")

    # ------- Containers -------
    weights = pd.DataFrame(0.0, index=rets.index, columns=rets.columns)
    prev_w = pd.Series(0.0, index=rets.columns)
    trade_rows = []

    # ------- Loop over rebalances -------
    for i, d in enumerate(rb_dates):
        end_loc = rets.index.get_loc(d)
        start_loc_window = max(0, end_loc - args.lookback)

        window_ret = (1.0 + rets.iloc[start_loc_window:end_loc]).prod() - 1.0
        window_ret = window_ret.replace([np.inf, -np.inf], np.nan).fillna(-1e9)

        # Rank and equal-weight the top-N candidates
        picks = window_ret.nlargest(args.top_n).index
        cand = pd.Series(0.0, index=rets.columns)
        if len(picks) > 0:
            cand.loc[picks] = 1.0 / len(picks)

        # Apply caps/min/max-holdings/cash buffer
        cand = cap_min_normalize(
            cand,
            max_holdings=args.max_holdings,
            wcap=float(args.weight_cap),
            wmin=float(args.min_weight),
            cash_buffer=float(args.cash_buffer),
        )

        # Turnover band decision
        if args.turnover_band > 0 and (cand - prev_w).abs().sum() <= args.turnover_band:
            w_new = prev_w.copy()
            changed = False
        else:
            w_new = cand
            changed = True

        # Block range for these weights
        start_block = end_loc
        end_block = len(rets)
        if i + 1 < len(rb_dates):
            end_block = rets.index.get_loc(rb_dates[i + 1])
        weights.iloc[start_block:end_block] = w_new.values

        # Transaction cost at boundary
        turnover = (w_new - prev_w).abs().sum()
        if start_block < len(rets):
            rets.iloc[start_block] = (
                rets.iloc[start_block] - (args.cost_bps / 1e4) * turnover
            )

        # Trades log
        block_end_date = (
            rets.index[end_block - 1]
            if end_block > start_block
            else rets.index[start_block]
        )
        for tkr in rets.columns:
            prev = float(prev_w.loc[tkr])
            new = float(w_new.loc[tkr])
            delta = new - prev
            action = "HOLD"
            if delta > 1e-12:
                action = "BUY"
            elif delta < -1e-12:
                action = "SELL"
            trade_rows.append(
                {
                    "date": d.date().isoformat(),
                    "block_start": rets.index[start_block].date().isoformat(),
                    "block_end": block_end_date.date().isoformat(),
                    "ticker": tkr,
                    "prev_w": prev,
                    "new_w": new,
                    "delta_w": delta,
                    "turnover_contrib": abs(delta),
                    "picked": int(tkr in picks),
                    "lookback_return": float(window_ret.loc[tkr]),
                    "action": action,
                    "weights_changed": int(changed),
                }
            )

        prev_w = w_new

    # ------- Portfolio returns -------
    port_ret = (weights * rets).sum(axis=1).fillna(0.0)

    # Mgmt fee (bps/yr) applied daily
    if args.mgmt_fee_bps != 0:
        fee_daily = (args.mgmt_fee_bps / 1e4) / 252.0
        port_ret = port_ret - fee_daily

    # Vol targeting (optional)
    if args.vol_target > 0:
        rolling_vol = port_ret.rolling(63).std().shift(1) * np.sqrt(252)
        target = args.vol_target / 100.0
        scale = np.where(rolling_vol > 1e-8, target / rolling_vol, 1.0)
        scale = np.clip(scale, 0.0, 3.0)
        port_ret = port_ret * pd.Series(scale, index=port_ret.index).fillna(1.0)

    eq = (1.0 + port_ret).cumprod()
    dd = eq / eq.cummax() - 1.0

    cagr = cagr_from_equity(eq)
    sharpe = (
        float(np.sqrt(252) * port_ret.mean() / port_ret.std())
        if port_ret.std() > 0
        else 0.0
    )
    maxdd = float(dd.min())
    calmar = float(cagr / abs(maxdd)) if maxdd < 0 else 0.0

    # ------- Optional benchmark -------
    bench_cagr = None
    bench_label = None
    bench_eq = None
    if args.benchmark.strip():
        bench_label = args.benchmark.strip()
        b_raw = yf.download(
            [bench_label],
            start=str(eq.index[0].date()),
            progress=False,
            auto_adjust=False,
        )
        b_adj = safe_adj_close(b_raw, [bench_label]).dropna()
        if not b_adj.empty:
            b_adj = b_adj.reindex(eq.index).ffill().dropna()
            b_ret = b_adj.pct_change().fillna(0.0)
            bench_eq = (1.0 + b_ret.squeeze()).cumprod()
            bench_cagr = cagr_from_equity(bench_eq)

    # ------- Filenames -------
    parts = [
        f"L{args.lookback}",
        f"K{args.top_n}",
        f"MH{args.max_holdings}",
        f"CAP{args.weight_cap:.2f}",
        f"MIN{args.min_weight:.2f}",
        f"CASH{args.cash_buffer:.2f}",
        f"RB{args.rebalance}",
        f"C{int(args.cost_bps)}",
        f"FEE{int(args.mgmt_fee_bps)}",
        f"VT{int(args.vol_target)}",
        f"TB{args.turnover_band:.2f}",
    ]
    stem = f"portfolioV2_{'-'.join([t.replace('.','_') for t in tickers])}_" + "_".join(
        parts
    )

    equity_csv = outdir / f"{stem}.csv"
    weights_csv = outdir / f"{stem}_weights.csv"
    trades_csv = outdir / f"{stem}_trades.csv"
    png = outdir / f"{stem}.png"

    # ------- Save -------
    pd.DataFrame({"equity": eq, "drawdown": dd, "port_ret": port_ret}).rename_axis(
        "date"
    ).to_csv(equity_csv, float_format="%.10f")

    weights.rename_axis("date").to_csv(weights_csv, float_format="%.6f")
    pd.DataFrame(trade_rows).to_csv(trades_csv, index=False)

    # ------- Plot -------
    plt.figure(figsize=(11, 5.5))
    ax = plt.gca()
    eq.plot(ax=ax, label="Strategy")
    if bench_eq is not None:
        bench_eq.plot(ax=ax, label=f"Benchmark ({bench_label})", alpha=0.75)
    ax.set_title(
        f"Equity | L={args.lookback}, K={args.top_n}, RB={args.rebalance}, "
        f"cap={args.weight_cap:.2f}, min={args.min_weight:.2f}, cash={args.cash_buffer:.2f}, "
        f"fee={args.mgmt_fee_bps:.0f}bps, VT={args.vol_target}%, TB={args.turnover_band}"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity (base=1.0)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(png, dpi=140)
    plt.close()

    # ------- JSON summary -------
    summary = {
        "csv": str(equity_csv),
        "weights_csv": str(weights_csv),
        "trades_csv": str(trades_csv),
        "png": str(png),
        "CAGR": cagr,
        "Sharpe": sharpe,
        "MaxDD": maxdd,
        "Calmar": calmar,
        "first_date": str(eq.index[0].date()),
        "last_date": str(eq.index[-1].date()),
        "points": int(len(eq)),
        "Benchmark": bench_label,
        "Benchmark_CAGR": bench_cagr,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

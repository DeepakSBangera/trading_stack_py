# tools/pilot_backtest.py
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FREQ_MAP = {"DE": "D", "WE": "W-FRI", "ME": "M", "QE": "Q", "YE": "A"}


def load_universe(csv_path: Path):
    if csv_path and csv_path.exists():
        df = pd.read_csv(csv_path)
        # expect "symbol" col containing Sxxx.SYN; fallback if not
        col = "symbol" if "symbol" in df.columns else df.columns[0]
        return [s.strip() for s in df[col].tolist()]
    return []


def load_close_matrix(parquet_root: Path, tickers: list[str]):
    px = {}
    for sym in tickers:
        stem = sym.replace(".SYN", "")
        fp = parquet_root / f"{stem}.parquet"
        if not fp.exists():
            continue
        df = pd.read_parquet(fp)
        if "close" not in df.columns:
            continue
        s = pd.Series(df["close"].values, index=pd.to_datetime(df.index), name=sym)
        px[sym] = s.asfreq("B").ffill()
    return pd.DataFrame(px).sort_index()


def rebalance_index(index: pd.DatetimeIndex, code: str):
    if code not in FREQ_MAP:
        code = "ME"
    return pd.Series(index=index, data=1).resample(FREQ_MAP[code]).last().index.intersection(index)


def apply_constraints(names, cap, min_w, target_sum, max_holdings=None):
    if max_holdings is not None:
        names = names[:max_holdings]
    if not names:
        return pd.Series(dtype=float)
    w = pd.Series(0.0, index=names, dtype=float)
    equal = max(min_w, target_sum / len(names))
    w.loc[names] = equal
    w = w.clip(upper=cap)
    if w.sum() > 0:
        w *= target_sum / w.sum()
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet-root", default="data/raw/synth/daily/SYN")
    ap.add_argument("--universe-csv", default="my_universe.csv")
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--lookback", type=int, default=126)
    ap.add_argument("--top-n", type=int, default=30)
    ap.add_argument("--rebalance", default="ME", help="DE|WE|ME|QE|YE")
    ap.add_argument("--weight-cap", type=float, default=0.07)
    ap.add_argument("--min-weight", type=float, default=0.01)
    ap.add_argument("--cash-buffer", type=float, default=0.03)
    ap.add_argument("--max-holdings", type=int, default=30)
    ap.add_argument("--cost-bps", type=float, default=15.0)
    ap.add_argument("--mgmt-fee-bps", type=float, default=75.0)
    ap.add_argument("--vol-target", type=float, default=0.0, help="annualized %, 0=off")
    ap.add_argument("--turnover-band", type=float, default=0.0, help="L1 threshold 0–2")
    ap.add_argument("--name", default="pilot_v1")
    ap.add_argument("--outdir", default="reports")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Universe
    tickers = load_universe(Path(args.universe_csv))
    if not tickers:
        # if universe file missing, infer from parquet directory
        tickers = [fp.stem + ".SYN" for fp in Path(args.parquet_root).glob("*.parquet")]
    if not tickers:
        print(json.dumps({"ok": False, "reason": "no universe or parquet files found"}, indent=2))
        return

    # Prices
    px = load_close_matrix(Path(args.parquet_root), tickers)
    px = px.loc[px.index >= pd.to_datetime(args.start)]
    if px.empty:
        print(json.dumps({"ok": False, "reason": "no price data after start"}, indent=2))
        return
    rets = px.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Rebalance dates
    rbd = rebalance_index(rets.index, args.rebalance)
    rbd = rbd.intersection(rets.index)
    if len(rbd) == 0:
        print(json.dumps({"ok": False, "reason": "no rebalance dates"}, indent=2))
        return

    # Backtest state
    weights = pd.DataFrame(0.0, index=rets.index, columns=rets.columns)
    prev_w = pd.Series(0.0, index=rets.columns)

    for i, d in enumerate(rbd):
        end = rets.index.get_loc(d)
        start_i = max(0, end - args.lookback)
        window_ret = (1.0 + rets.iloc[start_i:end]).prod() - 1.0
        picks = window_ret.nlargest(args.top_n).index.tolist()

        target_sum = 1.0 - args.cash_buffer
        w_cand = apply_constraints(
            picks, args.weight_cap, args.min_weight, target_sum, args.max_holdings
        )

        # stickiness via turnover band
        if (
            args.turnover_band > 0
            and (w_cand.reindex(prev_w.index).fillna(0) - prev_w).abs().sum() <= args.turnover_band
        ):
            w_use = prev_w.copy()
        else:
            w_use = prev_w.copy() * 0.0
            w_use.loc[w_cand.index] = w_cand.values

        start_loc = end
        end_loc = len(rets)
        if i + 1 < len(rbd):
            end_loc = rets.index.get_loc(rbd[i + 1])
        weights.iloc[start_loc:end_loc] = w_use.values

        # transaction costs at boundary
        turnover = (w_use - prev_w).abs().sum()
        if start_loc < len(rets):
            rets.iloc[start_loc] = rets.iloc[start_loc] - (args.cost_bps / 1e4) * turnover

        prev_w = w_use

    # Base portfolio returns
    port_ret = (weights * rets).sum(axis=1).fillna(0.0)

    # Mgmt fee (daily)
    if args.mgmt_fee_bps > 0:
        port_ret = port_ret - (args.mgmt_fee_bps / 1e4) / 252.0

    # Vol targeting (63d)
    if args.vol_target > 0:
        roll = port_ret.rolling(63).std().shift(1) * np.sqrt(252)
        target = args.vol_target / 100.0
        scale = np.where(roll > 1e-8, target / roll, 1.0)
        scale = np.clip(scale, 0.0, 3.0)
        port_ret = port_ret * pd.Series(scale, index=port_ret.index).fillna(1.0)

    eq = (1.0 + port_ret).cumprod()
    dd = eq / eq.cummax() - 1.0

    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    cagr = float(eq.iloc[-1] ** (1 / years) - 1.0) if eq.iloc[-1] > 0 else 0.0
    sharpe = float(np.sqrt(252) * port_ret.mean() / port_ret.std()) if port_ret.std() > 0 else 0.0
    maxdd = float(dd.min())
    calmar = float(cagr / abs(maxdd)) if maxdd < 0 else 0.0

    # Output
    stem = f"pilot_{args.name}_L{args.lookback}_K{args.top_n}_{args.rebalance}_C{int(args.cost_bps)}_F{int(args.mgmt_fee_bps)}_VT{int(args.vol_target)}_TB{args.turnover_band:.2f}"
    csv = Path(args.outdir) / f"{stem}.csv"
    png = Path(args.outdir) / f"{stem}.png"
    pd.DataFrame({"equity": eq, "drawdown": dd, "port_ret": port_ret}).rename_axis("date").to_csv(
        csv, float_format="%.10f"
    )

    plt.figure(figsize=(10, 5))
    eq.plot()
    plt.title(f"Pilot Equity ({args.rebalance}; L={args.lookback}; K={args.top_n})")
    plt.xlabel("Date")
    plt.ylabel("Equity (base=1.0)")
    plt.tight_layout()
    plt.savefig(png, dpi=140)
    plt.close()

    print(
        json.dumps(
            {
                "ok": True,
                "csv": str(csv),
                "png": str(png),
                "CAGR": cagr,
                "Sharpe": sharpe,
                "MaxDD": maxdd,
                "Calmar": calmar,
                "first_date": str(eq.index[0].date()),
                "last_date": str(eq.index[-1].date()),
                "points": int(len(eq)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

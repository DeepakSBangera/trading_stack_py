from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from pathlib import Path

import pandas as pd

# ----------------------------
# Config
# ----------------------------
TOP_N = 5  # pick top-N assets each rebalance
REBAL_FREQ = "ME"  # month-end; use "ME" (not "M") to avoid deprecation warnings
RUNS_DIR = Path("reports/backtests")

# try these columns in each parquet file (case-sensitive)
PRICE_CANDIDATES = ["adj close", "Adj Close", "close", "Close"]


# ----------------------------
# Loading prices
# ----------------------------
def load_prices_from_parquet(dir_path: Path) -> pd.DataFrame:
    """Load all *.parquet in dir into a single DataFrame of prices (DatetimeIndex)."""
    frames: list[pd.Series] = []
    problems: list[str] = []

    for p in sorted(dir_path.glob("*.parquet")):
        try:
            df = pd.read_parquet(p)

            # Ensure DatetimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                if "date" in df.columns:
                    df = df.set_index(pd.to_datetime(df["date"], errors="coerce")).drop(
                        columns=["date"]
                    )
                else:
                    # sometimes parquet saved the index as object; try coercion
                    try:
                        df.index = pd.to_datetime(df.index, errors="coerce")
                    except Exception:
                        pass

            # pick a price column
            price_col = None
            for c in PRICE_CANDIDATES:
                if c in df.columns:
                    price_col = c
                    break
            if price_col is None and df.shape[1] == 1:
                price_col = df.columns[0]

            if price_col is None:
                problems.append(f"{p.name}: no price-like column (had {list(df.columns)})")
                continue
            if not isinstance(df.index, pd.DatetimeIndex):
                problems.append(f"{p.name}: could not determine datetime index")
                continue

            s = pd.to_numeric(df[price_col], errors="coerce").rename(p.stem)
            s = s[~s.index.isna()].sort_index()
            if len(s):
                frames.append(s)
            else:
                problems.append(f"{p.name}: empty after cleaning")

        except Exception as e:
            problems.append(f"{p.name}: {type(e).__name__}: {e}")

    if not frames:
        hint = "\n".join(problems[:10])
        raise FileNotFoundError(
            f"No usable parquet series found in {dir_path}.\nSample diagnostics:\n{hint}"
        )

    px = pd.concat(frames, axis=1).sort_index()
    return px


# ----------------------------
# Momentum scoring & weights
# ----------------------------
def monthly_last(px: pd.DataFrame) -> pd.DataFrame:
    """Month-end prices."""
    return px.resample(REBAL_FREQ).last()


def r12_1_scores(px_m: pd.DataFrame) -> pd.DataFrame:
    """
    Momentum score = 12-month return minus 1-month return, computed on month-end prices.
    """
    r12 = px_m / px_m.shift(12) - 1.0
    r01 = px_m / px_m.shift(1) - 1.0
    return r12 - r01


def build_monthly_weights(px: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """
    Equal-weight the top_n assets by r12-1 score each month-end.
    Returns a DataFrame indexed by month-ends with weights per asset.
    """
    px_m = monthly_last(px)
    scores = r12_1_scores(px_m)

    w_m = pd.DataFrame(0.0, index=scores.index, columns=px.columns)

    for date, row in scores.iterrows():
        row = row.dropna()
        if not len(row):
            continue
        winners = row.nlargest(top_n).index[:top_n]
        if len(winners):
            w_m.loc[date, winners] = 1.0 / len(winners)

    # If there are leading months with all-zeros (before we have 12M history), keep as zeros.
    return w_m


# ----------------------------
# Backtest core
# ----------------------------
@dataclass
class BacktestResult:
    px: pd.DataFrame
    w_m: pd.DataFrame
    w_d: pd.DataFrame
    port_ret: pd.Series
    equity: pd.Series
    metrics: dict


def run_backtest(px: pd.DataFrame, top_n: int = TOP_N) -> BacktestResult:
    # Monthly target weights
    w_m = build_monthly_weights(px, top_n=top_n)

    # Daily weights: hold month-end target until next rebalance
    w_d = w_m.reindex(px.index, method="ffill").fillna(0.0)

    # Daily returns (avoid deprecated fill; then fill first row NAs with 0)
    ret = px.pct_change(fill_method=None).fillna(0.0)

    # Use yesterday's weights to earn today's returns
    port_ret = (w_d.shift(1).fillna(0.0) * ret).sum(axis=1)

    # Equity curve (start at 1.0)
    equity = (1.0 + port_ret).cumprod()

    # Metrics
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0)

    ann_ret = float((1.0 + port_ret.mean()) ** 252 - 1.0)
    ann_vol = float(port_ret.std(ddof=0) * sqrt(252))
    sharpe = float(ann_ret / ann_vol) if ann_vol > 0 else float("nan")

    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    mdd = float(dd.min())

    metrics = {
        "TOP_N": top_n,
        "REBAL_FREQ": REBAL_FREQ,
        "start": str(px.index[0]),
        "end": str(px.index[-1]),
        "CAGR": cagr,
        "AnnRet": ann_ret,
        "AnnVol": ann_vol,
        "Sharpe(0%)": sharpe,
        "MaxDrawdown": mdd,
    }

    return BacktestResult(
        px=px, w_m=w_m, w_d=w_d, port_ret=port_ret, equity=equity, metrics=metrics
    )


# ----------------------------
# Saving & plotting
# ----------------------------
def save_outputs(res: BacktestResult, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    # Inputs/outputs for reproducibility and downstream tooling
    res.px.to_csv(outdir / "px.csv")
    res.w_m.to_csv(outdir / "monthly_weights.csv")
    # 1-row, vertical view of the latest weights
    (res.w_m.tail(1).T.rename(columns=lambda _: "weight")).to_csv(outdir / "last_weights.csv")
    res.equity.to_csv(outdir / "equity_curve.csv", header=["equity"])

    # Optional plot if matplotlib is available
    try:
        import matplotlib.pyplot as plt  # type: ignore

        fig = plt.figure(figsize=(8, 4.5))
        res.equity.plot(title="Equity Curve")
        plt.xlabel("")
        plt.tight_layout()
        fig.savefig(outdir / "equity_curve.png", dpi=144)
        plt.close(fig)
    except ModuleNotFoundError as e:
        print(f"[info] Skipped plots: {e}")


# ----------------------------
# CLI
# ----------------------------
def main() -> None:
    root = Path(__file__).resolve().parents[1]
    prices_dir = root / "data" / "prices"

    print(f"Loading prices from: {prices_dir}")
    px = load_prices_from_parquet(prices_dir)
    print(
        f"Loaded {px.shape[1]} tickers × {px.shape[0]} rows "
        f"({px.index.min().date()} → {px.index.max().date()})."
    )

    res = run_backtest(px, top_n=TOP_N)

    # Pretty print summary
    print("\n=== Backtest summary ===")
    for k in [
        "TOP_N",
        "REBAL_FREQ",
        "start",
        "end",
        "CAGR",
        "AnnRet",
        "AnnVol",
        "Sharpe(0%)",
        "MaxDrawdown",
    ]:
        print(f"{k:>12}: {res.metrics[k]}")

    # Save run artifacts
    run_dir = RUNS_DIR / pd.Timestamp.now().strftime("%Y-%m-%d_%H%M")
    save_outputs(res, run_dir)
    print(f"\nSaved outputs to: {run_dir}")


if __name__ == "__main__":
    main()

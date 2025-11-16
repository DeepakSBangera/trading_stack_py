import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def infer_equity(df: pd.DataFrame) -> pd.Series:
    # explicit equity/cumreturn columns
    eq_cols = [
        "equity",
        "Equity",
        "cum_equity",
        "cum_equity_value",
        "cum_return",
        "cumreturn",
        "cumret",
        "CumulativeReturn",
        "portfolio_value",
        "nav",
        "NAV",
    ]
    for c in eq_cols:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().sum() > 3:
                # looks like returns? convert to equity
                if s.max() <= 2.0 and s.min() > -0.95 and (s.head(10).abs().max() < 0.6):
                    return (1.0 + s.fillna(0)).cumprod()
                return s

    # typical returns columns -> cumprod to equity
    for c in ["ret", "returns", "strategy_return", "strategy_ret", "portfolio_ret"]:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
            return (1.0 + s).cumprod()

    # fallback: normalize a price-like column
    price_like = [c for c in df.columns if c.lower() in ("close", "adjclose", "adj_close", "price")]
    if not price_like:
        numerics = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        price_like = numerics[:1] if numerics else []
    if price_like:
        s = pd.to_numeric(df[price_like[0]], errors="coerce")
        s = s / s.dropna().iloc[0]
        return s

    raise ValueError("Could not infer an equity series from CSV.")


def to_drawdown(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return equity / peak - 1.0


def estimate_years(eq: pd.Series) -> float:
    # If index is datetime-like, use actual span; else assume ~252 trading days/yr
    if isinstance(eq.index, pd.DatetimeIndex):
        delta_days = max((eq.index[-1] - eq.index[0]).days, 1)  # avoid zero
        return max(delta_days / 365.25, 1e-9)
    return max(len(eq) / 252.0, 1e-9)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--outdir", default="reports")
    p.add_argument("--title", default="")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)

    # best-effort date index if present
    for col in ["date", "Date", "timestamp", "Timestamp", "time", "Time"]:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce", utc=False)
                df = df.set_index(col)
                break
            except Exception:
                pass

    eq = infer_equity(df).astype(float).dropna().rename("equity")
    # normalize start near 1.0 for readability
    if eq.iloc[0] != 0 and not (0.8 <= float(eq.iloc[0]) <= 1.2):
        eq = eq / float(eq.iloc[0])

    dd = to_drawdown(eq).rename("drawdown")

    years = estimate_years(eq)
    total_ret = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    try:
        cagr = float(eq.iloc[-1] ** (1.0 / years) - 1.0)
    except Exception:
        cagr = float("nan")
    maxdd = float(dd.min())

    summary = {
        "csv": str(Path(args.csv).resolve()),
        "total_return": total_ret,
        "CAGR": cagr,
        "MaxDD": maxdd,
        "first_idx": str(eq.index[0]) if len(eq) > 0 else None,
        "last_idx": str(eq.index[-1]) if len(eq) > 0 else None,
        "points": int(eq.size),
        "years_est": years,
        "datetime_index": isinstance(eq.index, pd.DatetimeIndex),
    }

    title = args.title or Path(args.csv).stem
    stem = Path(args.csv).stem
    png_eq = outdir / f"{stem}_equity.png"
    png_dd = outdir / f"{stem}_drawdown.png"
    json_path = outdir / f"{stem}_summary.json"

    is_dt = isinstance(eq.index, pd.DatetimeIndex)
    xlab = "Date" if is_dt else "Bars"

    # equity
    plt.figure(figsize=(10, 4))
    plt.plot(eq.index, eq.values)
    plt.title(f"Equity curve — {title}")
    plt.xlabel(xlab)
    plt.ylabel("Equity (normalized)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(png_eq, dpi=150)
    plt.close()

    # drawdown
    plt.figure(figsize=(10, 3))
    plt.plot(dd.index, dd.values)
    plt.title(f"Drawdown — {title}")
    plt.xlabel(xlab)
    plt.ylabel("Drawdown")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(png_dd, dpi=150)
    plt.close()

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved:", png_eq)
    print("Saved:", png_dd)
    print("Saved:", json_path)
    print("Summary:", summary)


if __name__ == "__main__":
    main()

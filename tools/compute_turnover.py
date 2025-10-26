import argparse
import glob
import os

import numpy as np
import pandas as pd


def latest(pat: str):
    files = sorted(glob.glob(pat), key=os.path.getmtime, reverse=True)
    return files[0] if files else None


def load_table(path: str) -> pd.DataFrame:
    return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)


def pick_source(reports_dir: str):
    trades = latest(os.path.join(reports_dir, "portfolioV2_*_trades.parquet")) or latest(
        os.path.join(reports_dir, "portfolioV2_*_trades.csv")
    )
    weights = latest(os.path.join(reports_dir, "portfolioV2_*_weights.parquet")) or latest(
        os.path.join(reports_dir, "portfolioV2_*_weights.csv")
    )
    return trades, weights


def monthly_from_trades(df: pd.DataFrame):
    if "date" not in df.columns:
        return None
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    for c in df.columns:
        if c.lower() in ("weight_delta", "notional_delta", "shares_delta", "turnover", "qty_delta"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    cand = next(
        (
            c
            for c in ("notional_delta", "weight_delta", "shares_delta", "qty_delta", "turnover")
            if c in df.columns
        ),
        None,
    )
    if cand is None and {"weight_before", "weight_after"}.issubset(df.columns):
        df["weight_delta"] = pd.to_numeric(df["weight_after"], errors="coerce") - pd.to_numeric(
            df["weight_before"], errors="coerce"
        )
        cand = "weight_delta"
    if cand is None:
        return None
    df["ym"] = df["date"].dt.to_period("M").dt.to_timestamp()
    out = (
        df.groupby("ym")[cand]
        .apply(lambda s: s.abs().sum())
        .rename("monthly_turnover")
        .reset_index()
    )
    out["monthly_turnover"] = (
        pd.to_numeric(out["monthly_turnover"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    out["source"] = "trades"
    return out


def monthly_from_weights(path: str):
    w = load_table(path)
    if "date" not in w.columns:
        raise SystemExit("weights missing 'date'")
    date_col = "date"
    symbol_col = next(
        (
            c
            for c in w.columns
            if c.lower() in ("symbol", "ticker", "name", "secid") and c != date_col
        ),
        None,
    )
    weight_col = next(
        (
            c
            for c in w.columns
            if c.lower() in ("weight", "w", "target_weight", "final_weight") and c != date_col
        ),
        None,
    )
    if symbol_col is None or weight_col is None:
        non_date = [c for c in w.columns if c != date_col]
        if len(non_date) >= 2:
            symbol_col, weight_col = non_date[:2]
        else:
            raise SystemExit("weights missing symbol/weight")
    w[date_col] = pd.to_datetime(w[date_col], errors="coerce")
    w = w.dropna(subset=[date_col])
    mat = w.pivot_table(
        index=date_col, columns=symbol_col, values=weight_col, aggfunc="last"
    ).sort_index()
    mat = mat.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    dmat = mat.diff().abs().fillna(0.0)
    gross_half = dmat.sum(axis=1) * 0.5
    out = pd.DataFrame(
        {
            "ym": gross_half.index.to_period("M").to_timestamp(),
            "monthly_turnover": gross_half.values,
        }
    )
    out["source"] = "weights"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--outparquet", default=None)
    args = ap.parse_args()

    trades, weights = pick_source(args.reports)
    out, src = None, None
    if trades:
        tmp = monthly_from_trades(load_table(trades))
        if tmp is not None:
            out, src = tmp, trades
    if out is None and weights:
        out, src = monthly_from_weights(weights), weights
    if out is None:
        raise SystemExit("No usable trades/weights source found in reports/")

    outpq = args.outparquet or os.path.join(args.reports, "wk3_turnover_profile.parquet")
    out.to_parquet(outpq, index=False, compression="snappy")
    print(f"✓ Wrote {outpq}")
    print(f"Source: {src}")
    print(
        f"Method: {out.iloc[0]['source'] if 'source' in out.columns and len(out)>0 else 'unknown'}"
    )


if __name__ == "__main__":
    main()

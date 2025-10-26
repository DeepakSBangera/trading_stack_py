# tools/build_portfolio.py
import argparse
import json
from pathlib import Path

import pandas as pd


def load_factor(root: Path, name: str):
    p = root / "marts" / "factors" / f"factor={name}" / "full.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def zscore_rank(df, ascending=False):
    return df.rank(axis=1, pct=True, ascending=ascending)


def rank_and_weight(
    mom, vol, quality, top_n=30, weight_cap=0.07, min_w=0.01, cash_buffer=0.03, max_holdings=None
):
    score = (
        0.5 * zscore_rank(mom, False)
        + 0.3 * zscore_rank(quality, False)
        + 0.2 * zscore_rank(vol, True)
    )
    picks = score.apply(lambda row: row.nlargest(top_n).index.tolist(), axis=1)

    weights = []
    for dt, names in picks.items():
        if max_holdings is not None:
            names = names[:max_holdings]
        w = pd.Series(0.0, index=score.columns, dtype=float)
        if names:
            target = 1.0 - cash_buffer
            equal = max(min_w, target / len(names))
            w.loc[names] = equal
            w = w.clip(upper=weight_cap)
            if w.sum() > 0:
                w *= target / w.sum()
        weights.append((dt, w))
    W = pd.DataFrame({dt: w for dt, w in weights}).T
    return W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--top-n", type=int, default=30)
    ap.add_argument("--weight-cap", type=float, default=0.07)
    ap.add_argument("--min-weight", type=float, default=0.01)
    ap.add_argument("--cash-buffer", type=float, default=0.03)
    ap.add_argument("--max-holdings", type=int, default=None)
    ap.add_argument("--name", default="pilot_v1")
    args = ap.parse_args()

    root = Path(args.root)
    mom = load_factor(root, "mom126")
    vol = load_factor(root, "vol20")
    qual = load_factor(root, "quality")
    if any(df.empty for df in [mom, vol, qual]):
        print(
            json.dumps({"ok": False, "reason": "missing factors; run build_factors.py"}, indent=2)
        )
        return

    idx = mom.index.intersection(vol.index).intersection(qual.index)
    mom, vol, qual = mom.loc[idx], vol.loc[idx], qual.loc[idx]

    W = rank_and_weight(
        mom,
        vol,
        qual,
        args.top_n,
        args.weight_cap,
        args.min_weight,
        args.cash_buffer,
        args.max_holdings,
    )
    out = root / "marts" / "portfolios" / f"name={args.name}"
    out.mkdir(parents=True, exist_ok=True)
    p = out / "weights.parquet"
    W.to_parquet(p)
    print(
        json.dumps(
            {
                "ok": True,
                "weights": str(p),
                "dates": [str(idx.min().date()), str(idx.max().date())],
                "names": W.shape[1],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

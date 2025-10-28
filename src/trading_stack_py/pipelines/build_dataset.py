from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd

from trading_stack_py.cv.walkforward import WalkForwardCV
from trading_stack_py.features import feature_functions as F
from trading_stack_py.utils.returns import to_excess_returns


def make_dataset(
    df: pd.DataFrame,
    date_col: str | None,
    price_col: str | None,
    freq: str,
    rf: float,
    target_h: int,
    target_kind: str,
):
    # returns
    if "returns" in df.columns:
        r = pd.to_numeric(df["returns"], errors="coerce")
    else:
        if not price_col or price_col not in df.columns:
            raise ValueError("Provide --price-col or a 'returns' column.")
        p = pd.to_numeric(df[price_col], errors="coerce").ffill()
        r = p.pct_change()

    r = pd.Series(to_excess_returns(r.to_numpy(), rf=rf, freq=freq), index=df.index)

    # features (all past-only)
    X = pd.DataFrame(
        {
            "lag1": F.lag(r, 1),
            "lag5": F.lag(r, 5),
            "rmean5": F.rolling_mean(r, 5),
            "rstd20": F.rolling_std(r, 20),
            "z20": F.zscore(r, 20),
            "rsi14": F.rsi((1 + r.fillna(0)).cumprod()),  # RSI on reconstructed price
        }
    )

    # target: future k-day return
    fwd = r.rolling(window=target_h, min_periods=target_h).sum().shift(-target_h + 1)
    if target_kind == "raw":
        y = fwd
    elif target_kind == "sign":
        y = np.sign(fwd)
    else:
        raise ValueError("target_kind must be 'raw' or 'sign'")

    data = pd.concat([X, y.rename("target")], axis=1).dropna()
    if date_col and date_col in df.columns:
        data.insert(0, "date", pd.to_datetime(df[date_col]).iloc[data.index].to_numpy())
    return data


def main():
    ap = argparse.ArgumentParser(description="Build leakage-safe dataset for modeling.")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--date-col", default=None)
    ap.add_argument("--price-col", default=None)
    ap.add_argument("--rf", type=float, default=0.0)
    ap.add_argument("--freq", default="D", choices=["D", "B", "W", "M", "Q"])
    ap.add_argument("--target-h", type=int, default=5)
    ap.add_argument("--target-kind", default="raw", choices=["raw", "sign"])
    ap.add_argument("--train", type=int, default=378)
    ap.add_argument("--test", type=int, default=63)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--expanding", action="store_true")
    ap.add_argument("--embargo", type=int, default=0)
    ap.add_argument("--min-train", type=int, default=None)
    ap.add_argument("--tag", default="W6")
    ap.add_argument("--outdir", default="reports/W6")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.date_col and args.date_col in df.columns:
        df = df.sort_values(args.date_col).reset_index(drop=True)

    data = make_dataset(
        df,
        args.date_col,
        args.price_col,
        args.freq,
        args.rf,
        args.target_h,
        args.target_kind,
    )

    feat_cols = [c for c in data.columns if c not in ("date", "target")]
    X = data[feat_cols].to_numpy(dtype=float)
    y = data["target"].to_numpy(dtype=float)
    n = len(data)

    cv = WalkForwardCV(
        train_size=args.train,
        test_size=args.test,
        step_size=args.step,
        expanding=args.expanding,
        embargo=args.embargo,
        min_train_size=args.min_train,
    )

    ts = time.strftime("%Y%m%d_%H%M%S")
    root = os.path.join(args.outdir, f"{args.tag}_{ts}")
    os.makedirs(root, exist_ok=True)

    rows = []
    seg = 0
    for tr, te in cv.split(n):
        seg += 1
        seg_dir = os.path.join(root, f"segment_{seg:02d}")
        os.makedirs(seg_dir, exist_ok=True)
        np.save(os.path.join(seg_dir, "X_train.npy"), X[tr])
        np.save(os.path.join(seg_dir, "y_train.npy"), y[tr])
        np.save(os.path.join(seg_dir, "X_test.npy"), X[te])
        np.save(os.path.join(seg_dir, "y_test.npy"), y[te])

        rows.append(
            {
                "segment": seg,
                "train_start": int(tr[0]),
                "train_end": int(tr[-1]),
                "test_start": int(te[0]),
                "test_end": int(te[-1]),
                "train_len": len(tr),
                "test_len": len(te),
            }
        )

    pd.DataFrame(rows).to_csv(os.path.join(root, "segments.csv"), index=False)
    data.to_parquet(os.path.join(root, "full_dataset.parquet"), index=False)

    with open(os.path.join(root, "README.md"), "w", encoding="utf-8") as f:
        f.write("# W6 Dataset Builder\n\n")
        f.write(f"- Rows after dropna: **{n}**\n")
        f.write(f"- Features: {feat_cols}\n")
        f.write(
            f"- Target: **{args.target_kind}** over **{args.target_h}** period(s)\n"
        )
        f.write(
            f"- CV: train/test/step/expanding/embargo/min_train = "
            f"{args.train}/{args.test}/{args.step or args.test}/{args.expanding}/{args.embargo}/{args.min_train}\n"
        )

    print("Dataset written to:", root)


if __name__ == "__main__":
    main()

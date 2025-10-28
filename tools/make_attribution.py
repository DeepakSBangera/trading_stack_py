# tools/make_attribution.py
import argparse
import pathlib
import sys
import traceback

import numpy as np
import pandas as pd

from tradingstack.metrics.attribution import (
    align_weights_and_returns,
    build_returns_from_prices,
    contribution_by_ticker,
    group_contribution,
    pivot_weights,
)


def _err(msg: str):
    print(f"ERROR: {msg}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--weights", required=True, help="weights long parquet (date,ticker,weight)"
    )
    ap.add_argument(
        "--prices-root",
        required=True,
        help="root folder of per-ticker parquet price/return files",
    )
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument(
        "--equity", required=True, help="portfolio_v2.parquet (for parity check)"
    )
    ap.add_argument(
        "--mapping",
        required=False,
        default="",
        help="CSV mapping ticker->sector (optional)",
    )
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load inputs
        w_long = pd.read_parquet(args.weights)
        eq = pd.read_parquet(args.equity)

        # Build wide weights
        w_wide = pivot_weights(w_long)  # tz-naive date index, ticker columns
        tickers = list(w_wide.columns)

        # Build wide returns from prices
        r_wide = build_returns_from_prices(pathlib.Path(args.prices_root), tickers)

        # Align
        W, R = align_weights_and_returns(w_wide, r_wide)

        # Contributions and portfolio return
        contrib, port_ret = contribution_by_ticker(W, R)

        # Optional sector/group frame
        if args.mapping and pathlib.Path(args.mapping).exists():
            _contrib_sector = group_contribution(
                contrib, pathlib.Path(args.mapping), "sector"
            )
        else:
            _contrib_sector = pd.DataFrame(index=contrib.index)

        # Parity check against equity file (if it has _nav)
        eq = eq.copy()
        if "date" in eq.columns:
            # Normalize equity date
            eq["date"] = pd.to_datetime(
                eq["date"], errors="coerce", utc=True
            ).dt.tz_convert(None)
            eq = eq.dropna(subset=["date"]).set_index("date").sort_index()
        eq.index.name = "date"
        if "_nav" in eq.columns:
            nav = (
                pd.to_numeric(eq["_nav"], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .ffill()
                .fillna(1.0)
            )
            # reconstruct portfolio NAV from port_ret
            aligned = port_ret.reindex(nav.index).fillna(0.0)
            recon_nav = (1.0 + aligned).cumprod()
            # compare shapes & difference
            common = nav.index.intersection(recon_nav.index)
            diff = (
                nav.reindex(common).fillna(method="ffill") - recon_nav.reindex(common)
            ).abs()
            with open(out_dir / "attribution_parity.txt", "w", encoding="utf-8") as fh:
                fh.write(f"dates={len(common)}\n")
                fh.write(f"abs_diff_sum={diff.sum():.8f}\n")
                fh.write("note=abs_diff_sum close to 0 indicates parity\n")
        else:
            with open(out_dir / "attribution_parity.txt", "w", encoding="utf-8") as fh:
                fh.write(
                    "dates=0\nabs_diff_sum=0.00000000\nnote=_nav not found in equity\n"
                )

        # Write outputs
        contrib.to_parquet(out_dir / "attribution_ticker.parquet")
        port_ret.to_frame("port_ret").to_parquet(
            out_dir / "attribution_portfolio_returns.parquet"
        )

        print("OK attribution ->", out_dir)
        print("  wrote:", out_dir / "attribution_ticker.parquet")
        print("  wrote:", out_dir / "attribution_portfolio_returns.parquet")
        print("  wrote:", out_dir / "attribution_parity.txt")

    except Exception as e:
        # Rich diagnostics
        _err(repr(e))
        _err("Traceback:")
        traceback.print_exc()

        try:
            print("\n--- DIAG: weights head/dtypes ---")
            print(w_long.head().to_string(index=False))
            print(w_long.dtypes)
        except Exception:
            pass
        try:
            print("\n--- DIAG: equity head/dtypes ---")
            print(eq.head().to_string())
            print(eq.dtypes)
        except Exception:
            pass
        try:
            print("\n--- DIAG: wide weights index/cols ---")
            print(getattr(w_wide, "index", None)[:5])
            print(getattr(w_wide, "columns", None))
        except Exception:
            pass
        try:
            print("\n--- DIAG: wide returns index/cols ---")
            print(getattr(r_wide, "index", None)[:5])
            print(getattr(r_wide, "columns", None))
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()

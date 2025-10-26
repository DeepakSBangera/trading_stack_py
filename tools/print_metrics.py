# tools/print_metrics.py
import argparse
import math

import numpy as np
import pandas as pd

# local package (ensure PYTHONPATH points to repo root when running)
from tradingstack.io.equity import load_equity
from tradingstack.metrics.calmar import calmar_ratio
from tradingstack.metrics.drawdown import max_drawdown
from tradingstack.metrics.sharpe import sharpe_annual
from tradingstack.metrics.sortino import sortino_annual

# omega is optional
try:
    from tradingstack.metrics.omega import omega_ratio  # type: ignore

    _HAS_OMEGA = True
except Exception:
    _HAS_OMEGA = False


def fmt(x: float, nd=6) -> str:
    try:
        if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
            return "NA"
        return str(round(float(x), nd))
    except Exception:
        return "NA"


def main() -> None:
    ap = argparse.ArgumentParser(description="Print basic portfolio metrics.")
    ap.add_argument(
        "--equity",
        required=True,
        help="Path to portfolio parquet with columns: date, _nav (and/or ret)",
    )
    ap.add_argument("--out", help="Optional path to write a 1-row parquet with metrics")
    args = ap.parse_args()

    df = load_equity(args.equity)
    if "date" not in df.columns:
        raise ValueError("equity file missing 'date'")
    if "_nav" not in df.columns:
        # derive nav if only ret exists
        if "ret_net" in df.columns:
            nav = (1.0 + pd.to_numeric(df["ret_net"], errors="coerce").fillna(0.0)).cumprod()
            df["_nav"] = nav
        elif "ret_gross" in df.columns:
            nav = (1.0 + pd.to_numeric(df["ret_gross"], errors="coerce").fillna(0.0)).cumprod()
            df["_nav"] = nav
        else:
            raise ValueError("equity file missing _nav and returns (ret_net/ret_gross)")

    # clean series
    nav = (
        pd.to_numeric(df["_nav"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
        .fillna(1.0)
    )
    rets = nav.pct_change().fillna(0.0)

    out = {
        "rows": int(len(df)),
        "max_dd": float(max_drawdown(nav)),
        "sortino_annual": float(sortino_annual(rets)),
        "calmar": float(calmar_ratio(nav)),
        "sharpe_annual": float(sharpe_annual(rets)),
    }
    if _HAS_OMEGA:
        try:
            out["omega_ratio"] = float(omega_ratio(rets))
        except Exception:
            out["omega_ratio"] = float("nan")

    # print human-readable
    print("rows=", out["rows"])
    print("max_dd=", fmt(out["max_dd"]))
    print("sortino_annual=", fmt(out["sortino_annual"]))
    print("calmar=", fmt(out["calmar"]))
    print("sharpe_annual=", fmt(out["sharpe_annual"]))
    if "omega_ratio" in out:
        print("omega_ratio=", fmt(out["omega_ratio"]))

    # optional parquet
    if args.out:
        m = pd.DataFrame([out])
        m.to_parquet(args.out, index=False)


if __name__ == "__main__":
    main()

# scripts/w3_turnover.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class TurnoverResult:
    per_rebalance: pd.DataFrame
    summary: pd.Series


ROOT = Path(__file__).resolve().parents[1]
BT_DIR = ROOT / "reports" / "backtests"
OUT_W3 = ROOT / "reports"
OUT_W3.mkdir(parents=True, exist_ok=True)


def _latest_backtest_dir(base: Path) -> Path:
    runs = [d for d in base.iterdir() if d.is_dir()]
    if not runs:
        raise FileNotFoundError(f"No backtest runs found in {base}")
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0]


def _load_run(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      px: price panel (DatetimeIndex, columns=tickers)
      w_m: monthly target weights at each rebalance date
    """
    px_path = run_dir / "px.csv"
    wm_path = run_dir / "monthly_weights.csv"
    if not px_path.exists() or not wm_path.exists():
        raise FileNotFoundError(
            f"Expected files not found:\n  {px_path}\n  {wm_path}\n"
            f"Run a backtest first: python -m scripts.w2_backtest"
        )
    px = pd.read_csv(px_path, index_col=0, parse_dates=True)
    w_m = pd.read_csv(wm_path, index_col=0, parse_dates=True)
    # Ensure aligned columns (drop any missing tickers quietly)
    common = sorted(set(px.columns) & set(w_m.columns))
    px = px[common].sort_index()
    w_m = w_m[common].sort_index()
    return px, w_m


def _period_returns(
    px: pd.DataFrame, start_dt: pd.Timestamp, end_dt: pd.Timestamp
) -> pd.Series:
    """Total return per asset between (start_dt, end_dt], using last available prices."""
    # Price at start = last price at or before start_dt
    p0 = px.loc[:start_dt].tail(1).squeeze()
    # Price at end   = last price at or before end_dt
    p1 = px.loc[:end_dt].tail(1).squeeze()
    rets = (p1 / p0 - 1.0).fillna(0.0)
    return pd.to_numeric(rets, errors="coerce").fillna(0.0)


def compute_turnover(px: pd.DataFrame, w_m: pd.DataFrame) -> TurnoverResult:
    """
    Classic turnover per rebalance:
      1) Start-of-period (pre) weights = previous weights drifted by asset returns
      2) Turnover = 0.5 * sum_i | w_target_i - w_pre_i |
    """
    rebal_dates = list(w_m.index)
    if len(rebal_dates) < 2:
        raise ValueError("Need >= 2 rebalance dates to compute turnover.")

    rows = []
    # Initialize "previous" weights as the first row (no trade assumed at very first start)
    prev_w = w_m.iloc[0].fillna(0.0)

    for i in range(1, len(rebal_dates)):
        prev_dt = rebal_dates[i - 1]
        dt = rebal_dates[i]
        w_target = w_m.loc[dt].fillna(0.0)

        # Drift previous weights over (prev_dt, dt]
        r = _period_returns(px, prev_dt, dt)
        grown = prev_w * (1.0 + r)
        if grown.sum() == 0:
            w_pre = prev_w.copy()  # degenerate case
        else:
            w_pre = grown / grown.sum()

        # Turnover = 0.5 * L1 distance
        l1 = (w_target - w_pre).abs().sum()
        turnover = 0.5 * l1

        # Adds / drops diagnostics
        adds = ((w_target > 0) & (w_pre <= 0)).sum()
        drops = ((w_target <= 0) & (w_pre > 0)).sum()
        names_in = int((w_target > 0).sum())

        rows.append(
            {
                "date": dt,
                "turnover": float(turnover),
                "adds": int(adds),
                "drops": int(drops),
                "names_in": names_in,
            }
        )
        prev_w = w_target  # next period's "previous" weights are the new targets

    df = pd.DataFrame(rows).set_index("date")

    # Summary stats
    s = pd.Series(
        {
            "avg_turnover": float(df["turnover"].mean()),
            "median_turnover": float(df["turnover"].median()),
            "p95_turnover": float(df["turnover"].quantile(0.95)),
            "rebalances": int(len(df)),
        }
    )
    return TurnoverResult(per_rebalance=df, summary=s)


def main() -> None:
    run_dir = _latest_backtest_dir(BT_DIR)
    px, w_m = _load_run(run_dir)
    res = compute_turnover(px, w_m)

    # Save detailed per-rebalance turnover into the run folder too
    per_path = run_dir / "turnover_by_rebalance.csv"
    res.per_rebalance.to_csv(per_path)

    # Week-3 top-level report
    wk3_csv = OUT_W3 / "wk3_turnover_profile.csv"
    res.summary.to_frame("value").to_csv(wk3_csv)

    # Small JSON with limits you can tune
    limits = {
        "policy_caps": {
            "target_avg_turnover_per_rebalance": 0.25,  # 25% per rebalance
            "target_p95_turnover": 0.45,  # 45% at 95th percentile
        }
    }
    with open(OUT_W3 / "wk3_turnover_policy.json", "w", encoding="utf-8") as f:
        json.dump(limits, f, indent=2)

    print(f"[W3] Turnover per rebalance saved: {per_path}")
    print(f"[W3] Summary saved: {wk3_csv}")


if __name__ == "__main__":
    main()

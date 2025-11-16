from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import norm

ROOT = Path(r"F:\Projects\trading_stack_py")
REPORTS = ROOT / "reports"
OUT_WF = REPORTS / "wk5_walkforward_dsr.csv"
OUT_LOG = REPORTS / "canary_log.csv"


def open_win(p: Path):
    if sys.platform.startswith("win") and p.exists():
        os.startfile(p)


def sharpe_annual(ret: pd.Series, rf_daily: float = 0.0) -> float:
    r = ret.fillna(0.0) - rf_daily
    mu = r.mean() * 252.0
    sig = r.std(ddof=1) * (252.0**0.5)
    return float(mu / sig) if sig > 0 else 0.0


# Bailey–Lopez de Prado Deflated Sharpe Ratio (simplified)
def deflated_sharpe_ratio(
    sr: float, n: int, trials: int = 10, skew: float = 0.0, kurt: float = 3.0
) -> float:
    if n <= 1:
        return 0.0
    # expected max SR from noise (approx)
    gamma = 0.5772156649
    sr_max_exp = (1 - gamma) * norm.ppf(1 - 1 / trials) if trials > 1 else 0.0
    sr_adj = sr - sr_max_exp
    # variance adjustment for non-normality
    var_sr = (1 + 0.5 * sr**2 - skew * sr + (kurt - 3) / 4) / max(1, n - 1)
    z = sr_adj / max(1e-12, math.sqrt(var_sr))
    return float(norm.cdf(z))


def walk_forward_splits(
    dates: pd.DatetimeIndex, train_win: int, test_win: int, step: int = None
):
    step = step or test_win
    i = 0
    n = len(dates)
    while True:
        tr_start = i
        tr_end = i + train_win
        ts_end = tr_end + test_win
        if ts_end > n:
            break
        yield dates[tr_start:tr_end], dates[tr_end:ts_end]
        i += step


def main():
    pos_pq = REPORTS / "positions_daily.parquet"
    if not pos_pq.exists():
        raise SystemExit(
            "Missing reports\\positions_daily.parquet — run W3 bootstrap first."
        )

    pos = pd.read_parquet(pos_pq)
    port = (
        pos[["date", "port_value"]]
        .drop_duplicates()
        .sort_values("date")
        .assign(date=lambda d: pd.to_datetime(d["date"]))
        .set_index("date")
    )
    port["ret"] = port["port_value"].pct_change().fillna(0.0)

    dates = port.index
    # 60d train / 20d test makes sense for real; our sample is 20d, so use 10/5 for demo
    train_win = min(10, len(dates) // 2)
    test_win = min(5, max(1, len(dates) - train_win))

    rows = []
    trials = 12  # pretend 12 strategy variants to penalize selection
    for tr_idx, ts_idx in walk_forward_splits(dates, train_win, test_win):
        tr = port.loc[tr_idx, "ret"]
        ts = port.loc[ts_idx, "ret"]
        sr_in = sharpe_annual(tr)
        sr_out = sharpe_annual(ts)
        dsr_in = deflated_sharpe_ratio(sr_in, len(tr), trials=trials)
        dsr_out = deflated_sharpe_ratio(sr_out, len(ts), trials=trials)
        rows.append(
            {
                "train_start": tr_idx[0].date().isoformat(),
                "train_end": tr_idx[-1].date().isoformat(),
                "test_start": ts_idx[0].date().isoformat(),
                "test_end": ts_idx[-1].date().isoformat(),
                "sr_in": round(sr_in, 4),
                "sr_out": round(sr_out, 4),
                "dsr_in": round(dsr_in, 4),
                "dsr_out": round(dsr_out, 4),
                "promote": bool(dsr_out > 0.5 and sr_out > 0),  # simple gate
            }
        )

    wf = pd.DataFrame(rows)
    REPORTS.mkdir(parents=True, exist_ok=True)
    wf.to_csv(OUT_WF, index=False)

    canary = {
        "wf_rows": int(wf.shape[0]),
        "promote_count": int(wf["promote"].sum()) if not wf.empty else 0,
        "promote_rule": "dsr_out>0.5 and sr_out>0",
        "out_csv": str(OUT_WF),
    }
    with OUT_LOG.open("a", encoding="utf-8") as f:
        f.write("W5 walk-forward complete\n")
    print(json.dumps(canary, indent=2))

    open_win(OUT_WF)


if __name__ == "__main__":
    main()

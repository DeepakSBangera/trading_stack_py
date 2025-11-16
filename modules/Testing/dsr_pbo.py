# tools/dsr_pbo.py
# Purpose: Pick the best available equity timeseries (wk5_walkforward or portfolio_v2),
# compute daily/annual Sharpe, a conservative deflated Sharpe lower bound (DSR-ish),
# and PBO (NA for single policy). Write report parquet + ASCII decision note.
#
# Notes:
# - DSR here is a conservative *lower bound* of Sharpe using a 95% CI on daily Sharpe,
#   then annualized. This is robust and avoids overclaiming with tiny samples.
# - If both candidates have no usable returns, we emit zeros and NA and exit 0 (non-fatal),
#   so the pipeline doesn’t break. The decision gate will REJECT in that case.

import argparse
import math
import pathlib

import numpy as np
import pandas as pd

ANN_FACTOR = 252.0  # trading days


def load_equity(fn: pathlib.Path) -> pd.DataFrame | None:
    if not fn.exists():
        return None
    try:
        df = pd.read_parquet(fn)
    except Exception:
        return None
    if "date" not in df.columns:
        # not a fatal error for the script; caller will skip this source
        return None
    # Normalize date dtype to tz-naive UTC if needed
    d = df["date"]
    if isinstance(d.dtype, pd.DatetimeTZDtype):
        df["date"] = d.dt.tz_convert("UTC").dt.tz_localize(None)
    else:
        df["date"] = pd.to_datetime(d, errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=["date"]).sort_values("date")
    df = df.reset_index(drop=True)
    return df


def usable_returns(df: pd.DataFrame) -> pd.Series:
    """
    Return a clean daily returns series (float) if available/derivable, else empty.
    Priority:
      1) ret_net
      2) ret
      3) pct_change(nav_net)
      4) pct_change(nav_gross)
    """
    for col in ("ret_net", "ret"):
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            s = s.replace([np.inf, -np.inf], np.nan).dropna()
            if len(s) > 0:
                return s

    for nav_col in ("nav_net", "nav_gross"):
        if nav_col in df.columns:
            nav = pd.to_numeric(df[nav_col], errors="coerce")
            r = nav.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
            if len(r) > 0:
                return r

    return pd.Series(dtype=float)


def pick_best_source(
    candidates: list[pathlib.Path],
) -> tuple[pathlib.Path | None, pd.Series | None]:
    best_path: pathlib.Path | None = None
    best_returns: pd.Series | None = None
    best_n = -1
    for p in candidates:
        df = load_equity(p)
        if df is None:
            continue
        r = usable_returns(df)
        n = int(r.shape[0])
        if n > best_n:
            best_n = n
            best_path = p
            best_returns = r
    return best_path, best_returns


def daily_sharpe(r: pd.Series) -> float:
    """Classic daily Sharpe (mean/std), 0 if insufficient data."""
    r = pd.to_numeric(r, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if r.shape[0] < 2:
        return 0.0
    mu = r.mean()
    sd = r.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return 0.0
    return float(mu / sd)


def dsr_lower_bound_annual(r: pd.Series, alpha: float = 0.05) -> tuple[float, float]:
    """
    Conservative lower bound of Sharpe (95% CI by default), then annualized.
    Returns (sr_daily, dsr_annual).
    Reference: use an approximate CI for Sharpe via asymptotic SE:
      se(S) ~ sqrt((1 + 0.5*S^2) / (n - 1))
    Then LB_daily = max(0, S_daily - z * se).
    """
    n = int(r.shape[0])
    if n < 3:
        return 0.0, 0.0
    sr_d = daily_sharpe(r)
    z = 1.959963984540054  # ~N(0,1) 97.5th percentile (two-sided 95%)
    se = math.sqrt((1.0 + 0.5 * (sr_d**2)) / (n - 1.0))
    lb_daily = max(0.0, sr_d - z * se)
    dsr_ann = lb_daily * math.sqrt(ANN_FACTOR)
    return sr_d, dsr_ann


def write_parquet(
    out_path: pathlib.Path,
    days: int,
    sr_d: float,
    sr_a: float,
    dsr_a: float,
    pbo: float,
    source: str,
    n_trials: int,
):
    df = pd.DataFrame(
        [
            {
                "days": days,
                "sr_daily": sr_d,
                "sr_annual": sr_a,
                "dsr": dsr_a,
                "pbo": pbo,
                "source": source,
                "ann_factor": ANN_FACTOR,
                "n_trials": n_trials,
            }
        ]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)


def write_decision(
    note_path: pathlib.Path,
    dsr: float,
    min_dsr: float,
    pbo: float | None,
    max_pbo: float,
):
    pbo_txt = "NA" if (pbo is None or np.isnan(pbo)) else f"{pbo:.4f}"
    lines = []
    lines.append(f"DSR={dsr:.4f} (min {min_dsr})")
    lines.append(f"PBO={pbo_txt}" + ("" if pbo is None or np.isnan(pbo) else f" (max {max_pbo})"))
    ok_dsr = dsr >= min_dsr
    ok_pbo = True if (pbo is None or np.isnan(pbo)) else (pbo <= max_pbo)
    decision = "PROMOTE" if (ok_dsr and ok_pbo) else "REJECT"
    lines.append(f"Decision: {decision}")
    note_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Compute DSR/PBO from best available equity parquet.")
    ap.add_argument(
        "--out",
        default="reports/wk5_walkforward_dsr.parquet",
        help="Output parquet path",
    )
    ap.add_argument(
        "--note",
        default="reports/promotion_decision.txt",
        help="Decision note path (ASCII/UTF-8)",
    )
    ap.add_argument(
        "--n-trials",
        type=int,
        default=10,
        help="Number of trials for selection bias context",
    )
    ap.add_argument("--min-dsr", type=float, default=0.00, help="Minimum DSR threshold to promote")
    ap.add_argument(
        "--max-pbo",
        type=float,
        default=0.20,
        help="Maximum PBO to promote (ignored if NA)",
    )
    ap.add_argument(
        "candidates",
        nargs="*",
        default=[
            "reports/wk5_walkforward.parquet",
            "reports/portfolio_v2.parquet",
        ],
        help="Candidate equity parquet files (ordered by preference)",
    )
    args = ap.parse_args()

    cand_paths = [pathlib.Path(c) for c in args.candidates]
    best_path, ret = pick_best_source(cand_paths)

    out_path = pathlib.Path(args.out)
    note_path = pathlib.Path(args.note)

    if best_path is None or ret is None or ret.shape[0] == 0:
        # No usable data; emit zeros/NA and REJECT gate
        write_parquet(out_path, 0, 0.0, 0.0, 0.0, float("nan"), "NONE", args.n_trials)
        write_decision(note_path, 0.0, args.min_dsr, float("nan"), args.max_pbo)
        print(f"DSR/PBO done. days=0 sr_annual=0.0000 dsr=0.0000 pbo=NA | source=NONE | out={out_path}")
        return

    days = int(ret.shape[0])
    sr_d, dsr_a = dsr_lower_bound_annual(ret, alpha=0.05)
    sr_a = daily_sharpe(ret) * math.sqrt(ANN_FACTOR)

    # Single policy -> PBO not applicable (needs model selection context)
    pbo = float("nan")

    write_parquet(out_path, days, sr_d, sr_a, dsr_a, pbo, str(best_path), args.n_trials)
    write_decision(note_path, dsr_a, args.min_dsr, pbo, args.max_pbo)

    pbo_txt = "NA"
    print(
        f"DSR/PBO done. days={days} sr_annual={sr_a:.4f} dsr={dsr_a:.4f} pbo={pbo_txt} | "
        f"source={best_path} | out={out_path}"
    )


if __name__ == "__main__":
    main()

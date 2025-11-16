# scripts/w11_blend.py
"""
W11 — Alpha blending & correlation control.

Inputs: one or more sleeve-return CSVs with columns:
- date (or timestamp): trading date
- ret                : daily sleeve return (decimal, e.g., 0.003 for 0.3%)

Example:
    python scripts/w11_blend.py ^
      --inputs "reports/signals/momentum.csv,reports/signals/event.csv" ^
      --out "reports/wk11_alpha_blend.csv" ^
      --penalty 0.15

Output CSV columns:
- date
- blend_ret          (per-date blended return)
- blend_vol          (same value on every row; overall stdev of blend_ret)
- blend_sharpe       (same value on every row; annualized if --annualize)
- cor_penalty        (same value on every row; mean off-diagonal corr)
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

DATE_CANDS: Final[tuple[str, ...]] = ("date", "timestamp")
RET_CANDS: Final[tuple[str, ...]] = ("ret", "return", "daily_ret")


def _first(haystack: Iterable[str], wants: Iterable[str]) -> str | None:
    lower_to_orig = {str(c).lower(): c for c in haystack}
    for w in wants:
        if w in lower_to_orig:
            return lower_to_orig[w]
    return None


def _load_ret_csv(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    dcol = _first(df.columns, DATE_CANDS)
    rcol = _first(df.columns, RET_CANDS)
    if not rcol:
        raise ValueError(f"{path}: missing a return column named one of {RET_CANDS}")
    if dcol:
        df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
        df = df.dropna(subset=[dcol]).sort_values(dcol).set_index(dcol)
    s = pd.to_numeric(df[rcol], errors="coerce").dropna().astype(float)
    if isinstance(s.index, pd.DatetimeIndex) and not s.index.is_monotonic_increasing:
        s = s.sort_index()
    s.name = path.stem
    return s


def _de_corr_weights(R: pd.DataFrame, penalty: float) -> np.ndarray:
    """
    Simple correlation-aware reweighting:
    - start at equal weights
    - compute mean correlation per sleeve (rho_i)
    - raw weight_i = 1 / (1 + penalty * max(0, rho_i))
    - normalize to sum 1
    """
    k = R.shape[1]
    if k == 1:
        return np.array([1.0])

    C = R.corr().values
    rho_i = (C.sum(axis=1) - np.diag(C)) / np.maximum(k - 1, 1)
    rho_i = np.maximum(rho_i, 0.0)
    raw = 1.0 / (1.0 + penalty * rho_i)
    w = raw / raw.sum()
    return w


def main() -> None:
    ap = argparse.ArgumentParser(description="W11 — Alpha blending & correlation control")
    ap.add_argument(
        "--inputs",
        required=True,
        help="Comma-separated list of input CSVs (each with date & ret columns).",
    )
    ap.add_argument("--out", default="reports/wk11_alpha_blend.csv", help="Output CSV path")
    ap.add_argument(
        "--penalty",
        type=float,
        default=0.15,
        help="Correlation penalty intensity (0..1 typical).",
    )
    ap.add_argument(
        "--annualize",
        action="store_true",
        help="Annualize Sharpe using sqrt(252) if set.",
    )
    args = ap.parse_args()

    paths = [Path(p.strip()) for p in str(args.inputs).split(",") if p.strip()]
    if not paths:
        raise SystemExit("No inputs provided")

    series = []
    for p in paths:
        if not p.exists():
            raise SystemExit(f"Missing input: {p}")
        series.append(_load_ret_csv(p))

    # Align by inner join on dates
    R = pd.concat(series, axis=1, join="inner").dropna(how="any")
    if R.empty:
        raise SystemExit("No overlapping dates across inputs")

    # Penalized weights
    w = _de_corr_weights(R, penalty=args.penalty)

    # Blended returns
    blend_ret = R.values @ w
    blend = pd.Series(blend_ret, index=R.index, name="blend_ret")

    # Stats
    vol = float(np.std(blend.values, ddof=0))
    mean = float(np.mean(blend.values))
    sharpe = mean / vol if vol > 0 else 0.0
    if args.annualize:
        sharpe *= np.sqrt(252.0)

    # Correlation summary (mean off-diagonal)
    C = R.corr().values
    k = C.shape[0]
    if k > 1:
        off_diag = C[np.triu_indices(k, k=1)]
        rho_bar = float(np.mean(off_diag)) if off_diag.size else 0.0
    else:
        rho_bar = 0.0

    out = pd.DataFrame(
        {
            "date": blend.index,
            "blend_ret": blend.values,
            "blend_vol": vol,
            "blend_sharpe": sharpe,
            "cor_penalty": rho_bar,
        }
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(
        f"W11 wrote {out_path.as_posix()} with {len(out)} rows | "
        f"vol={vol:.6f} sharpe={sharpe:.2f} rho_bar={rho_bar:.3f}"
    )


if __name__ == "__main__":
    main()

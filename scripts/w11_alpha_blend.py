# scripts/w11_alpha_blend.py
from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

PRICE_CANDS: Final[tuple[str, ...]] = (
    "adj close",
    "adj_close",
    "close",
    "price",
    "adjclose",
)
DATE_CANDS: Final[tuple[str, ...]] = ("date", "timestamp", "dt")


# ---------- tiny helpers ----------
def _first_match(cands: Iterable[str], cols: Iterable[str]) -> str | None:
    lower = {str(c).lower(): c for c in cols}
    for w in cands:
        if w in lower:
            return lower[w]
    return None


def _coerce_dtindex(df: pd.DataFrame) -> pd.DataFrame:
    d = _first_match(DATE_CANDS, df.columns)
    if d:
        df = df.copy()
        df[d] = pd.to_datetime(df[d], errors="coerce")
        df = df.dropna(subset=[d]).sort_values(d).set_index(d)
    return df


def _load_symbol_returns(csv_path: Path) -> pd.Series:
    """Load a single price CSV → daily returns (% change)."""
    df = pd.read_csv(csv_path)
    df = _coerce_dtindex(df)
    pcol = _first_match(PRICE_CANDS, df.columns)
    if not pcol:
        raise ValueError(f"{csv_path}: no price-like column")
    s = pd.to_numeric(df[pcol], errors="coerce").dropna()
    if isinstance(s.index, pd.DatetimeIndex) and not s.index.is_monotonic_increasing:
        s = s.sort_index()
    r = s.pct_change().dropna()
    r.name = csv_path.stem.upper()
    return r


def _load_symbol_panel(price_glob: str) -> pd.DataFrame:
    """Join all symbol returns into one DataFrame (index=date)."""
    rets = []
    for p in sorted(Path().glob(price_glob)):
        try:
            rets.append(_load_symbol_returns(p))
        except Exception:
            # Skip symbols that don’t parse; we just need enough breadth
            continue
    if not rets:
        raise FileNotFoundError(f"No usable price files matched {price_glob}")
    panel = pd.concat(rets, axis=1).dropna(how="all")
    return panel


def _load_sleeves(sleeve_glob: str) -> pd.DataFrame | None:
    """
    Each sleeve CSV should have a date-like column + one return column:
      - either a column literally named 'ret' / 'return'
      - or we take the first numeric column after the date
    Returned frame has columns = sleeve names (file stems), index=date.
    """
    paths = sorted(Path().glob(sleeve_glob))
    if not paths:
        return None

    out = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            df = _coerce_dtindex(df)
            # pick return column
            ret_col = None
            for k in ("ret", "return"):
                if _first_match((k,), df.columns):
                    ret_col = _first_match((k,), df.columns)
                    break
            if ret_col is None:
                # take the first numeric col that isn’t the index
                for c in df.columns:
                    if pd.api.types.is_numeric_dtype(df[c]):
                        ret_col = c
                        break
            if ret_col is None:
                continue
            s = pd.to_numeric(df[ret_col], errors="coerce").dropna()
            s.name = p.stem
            out.append(s)
        except Exception:
            continue

    if not out:
        return None
    sleeves = pd.concat(out, axis=1).dropna(how="all")
    return sleeves


# ---------- allocator ----------
def _ridge_mv_weights(
    R: pd.DataFrame, lam: float = 10.0, clip01: bool = True
) -> pd.Series:
    """
    Mean-variance style weights: w ∝ (Σ + λI)^{-1} μ
    - R: T×N returns
    - lam: ridge level (stabilizes Σ)
    - clip01: clip to [0,1] after solving, then renormalize
    """
    # drop all-nan columns
    R = R.dropna(axis=1, how="all").fillna(0.0)
    if R.shape[1] == 0:
        raise ValueError("No usable series in R")

    mu = R.mean().values  # (N,)
    Σ = np.cov(R.values, rowvar=False)  # (N,N)
    N = Σ.shape[0]
    Σ_ridge = Σ + lam * np.eye(N)

    try:
        w = np.linalg.solve(Σ_ridge, mu)  # raw
    except np.linalg.LinAlgError:
        # last-resort: pseudo-inverse
        w = np.linalg.pinv(Σ_ridge) @ mu

    w = np.maximum(w, 0.0) if clip01 else w
    if w.sum() <= 0:
        # fall back to equal-weight if degenerate
        w = np.ones_like(w) / len(w)
    else:
        w = w / w.sum()

    return pd.Series(w, index=R.columns, name="weight")


def _metrics(blend: pd.Series) -> tuple[float, float]:
    """Return (ann_ret, ann_vol) with 252d convention."""
    if len(blend) < 2:
        return (float("nan"), float("nan"))
    daily_mu = float(blend.mean())
    daily_sd = float(blend.std(ddof=1))
    ann_ret = daily_mu * 252.0
    ann_vol = daily_sd * (252.0**0.5)
    return (ann_ret, ann_vol)


def _avg_offdiag_corr(R: pd.DataFrame) -> float:
    R = R.dropna(axis=1, how="all").fillna(0.0)
    if R.shape[1] < 2:
        return 0.0
    C = np.corrcoef(R.values, rowvar=False)
    n = C.shape[0]
    if n < 2:
        return 0.0
    # average absolute off-diagonal correlation
    mask = ~np.eye(n, dtype=bool)
    return float(np.mean(np.abs(C[mask])))


# ---------- main ----------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="W11 — Alpha blending with correlation penalty"
    )
    ap.add_argument(
        "--out",
        default="reports/wk11_alpha_blend.csv",
        help="Output CSV (single-row summary)",
    )
    ap.add_argument(
        "--price-glob",
        default="data/csv/*.csv",
        help="Fallback: use symbols if no sleeves",
    )
    ap.add_argument(
        "--sleeve-glob",
        default="reports/sleeves/*.csv",
        help="Optional: sleeve return CSVs",
    )
    ap.add_argument(
        "--lambda", dest="lam", type=float, default=10.0, help="Ridge λ in (Σ + λI)"
    )
    ap.add_argument(
        "--window", type=int, default=252, help="Lookback window (days) for metrics"
    )
    args = ap.parse_args()

    # Prefer sleeves if available; else build from symbol returns
    sleeves = _load_sleeves(args.sleeve_glob)
    if sleeves is not None and sleeves.shape[1] >= 2:
        R = sleeves.copy()
        source = "sleeves"
    else:
        panel = _load_symbol_panel(args.price_glob)
        R = panel.copy()
        source = "symbols"

    # recent window for stability (if enough data)
    if args.window and len(R) > args.window:
        Rw = R.iloc[-args.window :]
    else:
        Rw = R

    # weights & blended series
    w = _ridge_mv_weights(Rw, lam=args.lam, clip01=True)
    # align and compute daily blend
    R_align = Rw[w.index]
    blend = (R_align * w.values).sum(axis=1)

    ann_ret, ann_vol = _metrics(blend)
    sharpe = float("nan") if ann_vol == 0 or np.isnan(ann_vol) else ann_ret / ann_vol
    cor_pen = _avg_offdiag_corr(Rw)

    out = pd.DataFrame(
        [
            {
                "date": (
                    Rw.index.max().date()
                    if isinstance(Rw.index, pd.DatetimeIndex)
                    else None
                ),
                "blend_ret": round(ann_ret / 252.0, 6),  # report daily expected return
                "blend_vol": round(ann_vol / (252.0**0.5), 6),  # report daily vol
                "blend_sharpe": round(sharpe, 3) if np.isfinite(sharpe) else np.nan,
                "cor_penalty": round(cor_pen, 3),
                "source": source,
                "n_series": int(Rw.shape[1]),
            }
        ]
    )

    # write
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"W11 alpha blend → {out_path.as_posix()} (source={source}, N={Rw.shape[1]})")


if __name__ == "__main__":
    main()

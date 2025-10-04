# scripts/w11_alpha_blend.py
from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

PRICE_CANDS = ("adj close", "adj_close", "close", "price")
DATE_CANDS = ("date", "timestamp", "dt")


def _first_match(cands: Iterable[str], cols: Iterable[str]) -> str | None:
    lu = {str(c).lower(): c for c in cols}
    for w in cands:
        if w in lu:
            return lu[w]
    return None


def load_price_series(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    dcol = _first_match(DATE_CANDS, df.columns)
    if dcol:
        df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
        df = df.dropna(subset=[dcol]).sort_values(dcol).set_index(dcol)
    pcol = _first_match(PRICE_CANDS, df.columns)
    if not pcol:
        raise ValueError(f"{path}: no price-like column")
    s = pd.to_numeric(df[pcol], errors="coerce").dropna()
    if isinstance(s.index, pd.DatetimeIndex) and not s.index.is_monotonic_increasing:
        s = s.sort_index()
    return s.astype(float)


def ew_blend_from_folder(glob_pat: str) -> tuple[pd.Series, pd.DataFrame]:
    """Return equal-weight daily portfolio returns and the aligned return panel."""
    series = []
    for p in sorted(Path().glob(glob_pat)):
        try:
            s = load_price_series(p)
            if len(s) >= 30:
                series.append((p.stem.upper(), s))
        except Exception:
            # skip files that don't parse
            continue

    if not series:
        raise SystemExit(f"No usable price files found under {glob_pat}")

    # Daily simple returns
    rets = []
    for sym, s in series:
        r = s.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        r.name = sym
        rets.append(r)

    panel = pd.concat(rets, axis=1, join="inner").dropna(how="any")
    if panel.empty or panel.shape[0] < 30:
        raise SystemExit("Not enough overlapping return history to compute blend")

    # Equal-weight portfolio
    ew = panel.mean(axis=1)
    return ew, panel


def summarize_blend(ew: pd.Series, panel: pd.DataFrame) -> dict[str, float | str]:
    # Annualization (approx, trading days)
    ann = 252.0
    mu = float(ew.mean()) * ann
    vol = float(ew.std(ddof=0)) * np.sqrt(ann)
    sharpe = float(mu / vol) if vol > 0 else 0.0

    # Average pairwise correlation as a crude correlation penalty
    if panel.shape[1] > 1:
        corr = panel.corr().to_numpy()
        n = corr.shape[0]
        # mean of off-diagonal entries
        offdiag = corr[np.triu_indices(n, k=1)]
        cor_penalty = float(np.nanmean(offdiag))
    else:
        cor_penalty = 0.0

    last_dt = ew.index.max()
    date_str = last_dt.strftime("%Y-%m-%d") if hasattr(last_dt, "strftime") else str(last_dt)

    return {
        "date": date_str,
        "blend_ret": round(mu, 6),
        "blend_vol": round(vol, 6),
        "blend_sharpe": round(sharpe, 4),
        "cor_penalty": round(cor_penalty, 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="W11 — Alpha blending & correlation control (baseline)"
    )
    ap.add_argument("--data-glob", default="data/csv/*.csv", help="Input price CSV glob")
    ap.add_argument("--out", default="reports/wk11_alpha_blend.csv", help="Output CSV path")
    args = ap.parse_args()

    ew, panel = ew_blend_from_folder(args.data_glob)
    row = summarize_blend(ew, panel)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(out, index=False)
    print(f"W11: wrote {out.as_posix()} with 1 row")


if __name__ == "__main__":
    main()

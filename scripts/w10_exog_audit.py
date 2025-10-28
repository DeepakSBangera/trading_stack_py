# scripts/w10_exog_audit.py
from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

DATE_CANDS: tuple[str, ...] = ("date", "timestamp", "dt")
PRICE_CANDS: tuple[str, ...] = ("adj close", "adj_close", "adjclose", "close", "price")


def pick(cols: Iterable[str], cands: Iterable[str]) -> str | None:
    lower = {str(c).lower(): c for c in cols}
    for w in cands:
        if w in lower:
            return lower[w]
    return None


def load_price_series(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    dcol = pick(df.columns, DATE_CANDS)
    if dcol:
        df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
        df = df.dropna(subset=[dcol]).sort_values(dcol).set_index(dcol)

    pcol = pick(df.columns, PRICE_CANDS)
    if not pcol:
        raise ValueError(f"{path}: no price-like column")

    s = pd.to_numeric(df[pcol], errors="coerce").dropna()
    if isinstance(s.index, pd.DatetimeIndex):
        s = s.sort_index()
    return s


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Audit overlap between prices and exogenous CSV."
    )
    ap.add_argument("--data-glob", default="data/csv/*.csv")
    ap.add_argument("--exog-csv", required=True)
    ap.add_argument("--out", default="reports/w10_exog_overlap_audit.csv")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ex = pd.read_csv(args.exog_csv)
    ex_dcol = pick(ex.columns, DATE_CANDS) or "date"
    ex[ex_dcol] = pd.to_datetime(ex[ex_dcol], errors="coerce")
    ex = ex.dropna(subset=[ex_dcol]).sort_values(ex_dcol)
    ex_min = ex[ex_dcol].min() if not ex.empty else pd.NaT
    ex_max = ex[ex_dcol].max() if not ex.empty else pd.NaT

    rows: list[dict[str, object]] = []
    for p in Path().glob(args.data_glob):
        try:
            s = load_price_series(p)
        except Exception as e:  # noqa: BLE001
            rows.append(
                {
                    "symbol": p.stem.upper(),
                    "y_min": "",
                    "y_max": "",
                    "exog_min": str(ex_min) if pd.notna(ex_min) else "",
                    "exog_max": str(ex_max) if pd.notna(ex_max) else "",
                    "overlap_days": 0,
                    "error": str(e),
                }
            )
            continue

        if isinstance(s.index, pd.DatetimeIndex) and not s.empty:
            y_min = s.index.min()
            y_max = s.index.max()
        else:
            y_min = pd.NaT
            y_max = pd.NaT

        if pd.isna(y_min) or pd.isna(y_max) or pd.isna(ex_min) or pd.isna(ex_max):
            overlap = 0
        else:
            lo = max(y_min, ex_min)
            hi = min(y_max, ex_max)
            overlap = max((hi - lo).days + 1, 0) if hi >= lo else 0

        rows.append(
            {
                "symbol": p.stem.upper(),
                "y_min": str(y_min.date()) if pd.notna(y_min) else "",
                "y_max": str(y_max.date()) if pd.notna(y_max) else "",
                "exog_min": str(ex_min.date()) if pd.notna(ex_min) else "",
                "exog_max": str(ex_max.date()) if pd.notna(ex_max) else "",
                "overlap_days": int(overlap),
            }
        )

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Wrote {out_path.as_posix()} ({len(rows)} rows)")


if __name__ == "__main__":
    main()

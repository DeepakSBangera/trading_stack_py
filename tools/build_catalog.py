#!/usr/bin/env python
"""
build_catalog.py
Indexes your parquet files and writes a compact catalog to:
    data/ref/catalog.csv

Supports:
- FLAT layout  : data_synth/prices/*.parquet (symbol from filename)
                 data_synth/fundamentals/*.parquet
- HIVE layout  : data/raw/{vendor}/{dataset}/symbol=SYMB/year=YYYY/*.parquet

Output columns:
    dataset, vendor, layout, path, symbol, year, min_date, max_date, rows
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Try PyArrow for fast single-column reads; fall back to pandas if missing.
try:
    import pyarrow.parquet as pq  # type: ignore
except Exception:
    pq = None  # noqa: F401

REPO_ROOT = Path(".").resolve()
OUT_DIR = REPO_ROOT / "data" / "ref"
OUT_PATH = OUT_DIR / "catalog.csv"

COLUMNS = [
    "dataset",  # equity_eod | fundamentals
    "vendor",  # synth | yahoo | other
    "layout",  # flat | hive
    "path",  # repo-relative path to parquet
    "symbol",
    "year",
    "min_date",
    "max_date",
    "rows",
]


def read_date_stats(parquet_path: Path) -> tuple[str | None, str | None, int | None]:
    """
    Return (min_date, max_date, row_count) using only the 'date' column when possible.
    Works even if 'date' column is missing (returns min/max None).
    """
    try:
        if pq is not None:
            # Fast path with Arrow
            table = pq.read_table(parquet_path, columns=["date"])
            nrows = table.num_rows
            try:
                col = table.column("date")
                if col.null_count == col.length:
                    return (None, None, nrows)
                # Some Arrow builds need conversion to pandas for min/max
                s = table.to_pandas()["date"]
                s = pd.to_datetime(s, errors="coerce").dropna()
                if s.empty:
                    return (None, None, nrows)
                return (str(s.min().date()), str(s.max().date()), nrows)
            except Exception:
                # If the column is missing or weird, just return row count
                return (None, None, nrows)
        else:
            # Fallback: pandas engine
            df = pd.read_parquet(parquet_path)
            nrows = len(df)
            if "date" not in df.columns:
                return (None, None, nrows)
            s = pd.to_datetime(df["date"], errors="coerce").dropna()
            if s.empty:
                return (None, None, nrows)
            return (str(s.min().date()), str(s.max().date()), nrows)
    except Exception:
        return (None, None, None)


def kv_from_parts(parts: list[str]) -> dict[str, str]:
    """Extract hive-style keys from segments like 'symbol=RELIANCE.NS'."""
    out: dict[str, str] = {}
    for p in parts:
        if "=" in p:
            k, v = p.split("=", 1)
            out[k] = v
    return out


def add_row(
    rows: list[dict],
    *,
    dataset: str,
    vendor: str,
    layout: str,
    path_rel: str,
    symbol: str | None,
    year: str | None,
    min_d: str | None,
    max_d: str | None,
    nrows: int | None,
) -> None:
    rows.append(
        {
            "dataset": dataset,
            "vendor": vendor,
            "layout": layout,
            "path": path_rel,
            "symbol": symbol,
            "year": year,
            "min_date": min_d,
            "max_date": max_d,
            "rows": nrows,
        }
    )


def scan_flat_prices(base: Path, vendor: str, rows: list[dict]) -> None:
    # e.g., data_synth/prices/*.parquet
    if not base.exists():
        return
    for p in sorted(base.glob("*.parquet")):
        symbol = p.stem  # RELIANCE.NS.parquet -> RELIANCE.NS
        min_d, max_d, nrows = read_date_stats(p)
        add_row(
            rows,
            dataset="equity_eod",
            vendor=vendor,
            layout="flat",
            path_rel=str(p.relative_to(REPO_ROOT).as_posix()),
            symbol=symbol,
            year=None,
            min_d=min_d,
            max_d=max_d,
            nrows=nrows,
        )


def scan_flat_fundamentals(base: Path, vendor: str, rows: list[dict]) -> None:
    # e.g., data_synth/fundamentals/*.parquet
    if not base.exists():
        return
    for p in sorted(base.glob("*.parquet")):
        min_d, max_d, nrows = read_date_stats(p)
        add_row(
            rows,
            dataset="fundamentals",
            vendor=vendor,
            layout="flat",
            path_rel=str(p.relative_to(REPO_ROOT).as_posix()),
            symbol=None,
            year=None,
            min_d=min_d,
            max_d=max_d,
            nrows=nrows,
        )


def scan_hive(base: Path, dataset: str, vendor: str, rows: list[dict]) -> None:
    # e.g., data/raw/yahoo/equity_eod/symbol=RELIANCE.NS/year=2024/part.parquet
    if not base.exists():
        return
    for p in base.rglob("*.parquet"):
        rel = p.relative_to(REPO_ROOT).as_posix()
        parts = p.relative_to(base).as_posix().split("/")
        meta = kv_from_parts(parts)
        symbol = meta.get("symbol")
        year = meta.get("year")
        min_d, max_d, nrows = read_date_stats(p)
        add_row(
            rows,
            dataset=dataset,
            vendor=vendor,
            layout="hive",
            path_rel=rel,
            symbol=symbol,
            year=year,
            min_d=min_d,
            max_d=max_d,
            nrows=nrows,
        )


def main() -> None:
    rows: list[dict] = []

    # Current flat layout you already use
    scan_flat_prices(REPO_ROOT / "data_synth" / "prices", vendor="synth", rows=rows)
    scan_flat_fundamentals(
        REPO_ROOT / "data_synth" / "fundamentals", vendor="synth", rows=rows
    )

    # Future hive-style layout (safe if missing)
    scan_hive(
        REPO_ROOT / "data" / "raw" / "yahoo" / "equity_eod",
        dataset="equity_eod",
        vendor="yahoo",
        rows=rows,
    )
    scan_hive(
        REPO_ROOT / "data" / "raw" / "yahoo" / "fundamentals",
        dataset="fundamentals",
        vendor="yahoo",
        rows=rows,
    )

    # Ensure output dir and write CSV (even if empty)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=COLUMNS)
    if not df.empty:
        sort_cols = [
            c
            for c in ["dataset", "vendor", "symbol", "year", "path"]
            if c in df.columns
        ]
        df = df.sort_values(sort_cols, na_position="last")
    df.to_csv(OUT_PATH, index=False)
    print(f"✓ Wrote {OUT_PATH.as_posix()}  (files indexed: {len(df)})")


if __name__ == "__main__":
    main()

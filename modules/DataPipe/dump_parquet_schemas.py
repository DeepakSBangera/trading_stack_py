from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TARGETS = [
    "reports/portfolio_v2.parquet",
    "reports/weights_v2_norm.parquet",
    "reports/attribution_ticker.parquet",
    "reports/attribution_portfolio_returns.parquet",
    "reports/rolling_metrics.parquet",
]

# ---------- JSON conversion helpers ----------


def _is_nan(x: Any) -> bool:
    try:
        return isinstance(x, float) and math.isnan(x)
    except Exception:
        return False


def to_jsonable(x: Any) -> Any:
    """Convert common pandas/numpy/scalar types to JSON-safe Python values."""
    # pandas Timestamps / Timedeltas
    if isinstance(x, pd.Timestamp):
        # None for NaT; otherwise ISO 8601
        return None if pd.isna(x) else x.isoformat()
    if isinstance(x, pd.Timedelta):
        return None if pd.isna(x) else x.isoformat()

    # numpy scalar (np.int64, np.float64, etc.)
    if isinstance(x, (np.generic,)):
        try:
            return x.item()
        except Exception:
            return str(x)

    # pathlib
    if isinstance(x, Path):
        return str(x)

    # floats NaN/Inf handling
    if isinstance(x, float):
        if math.isinf(x):
            return "Infinity" if x > 0 else "-Infinity"
        if math.isnan(x):
            return None

    # plain containers: recurse
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [to_jsonable(v) for v in x]

    return x


def make_json(obj: Any) -> str:
    return json.dumps(to_jsonable(obj), indent=2, ensure_ascii=False)


# ---------- schema sniffing ----------


def sniff_schema(path: Path) -> dict:
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        return {"path": str(path), "error": repr(e)}

    # columns (name + dtype string)
    cols = [{"name": c, "dtype": str(df[c].dtype)} for c in df.columns]

    # index info + small sample
    idx_name = str(df.index.name) if df.index.name is not None else None
    try:
        idx_dtype = str(df.index.dtype)
    except Exception:
        idx_dtype = None
    index_preview = [to_jsonable(v) for v in list(df.index[:3])]

    # small data head (converted)
    head_records = df.head(3).to_dict(orient="records")
    head_records = [to_jsonable(rec) for rec in head_records]

    return {
        "path": str(path),
        "rows": int(len(df)),
        "index": {"name": idx_name, "dtype": idx_dtype, "preview": index_preview},
        "columns": cols,
        "head": head_records,
    }


def main():
    root = Path(".").resolve()
    out = []
    for rel in TARGETS:
        p = (root / rel).resolve()
        if p.exists():
            out.append(sniff_schema(p))
        else:
            out.append({"path": str(p), "missing": True})
    print(make_json(out))


if __name__ == "__main__":
    main()

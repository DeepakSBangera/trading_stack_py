import pathlib as P
import sys

import pandas as pd

fn = sys.argv[1] if len(sys.argv) > 1 else None
if not fn:
    print("usage: python peek_cols.py <parquet-file>")
    sys.exit(2)

p = P.Path(fn)
try:
    df = pd.read_parquet(p)
except Exception as e:
    print("READ_ERROR:", e)
    sys.exit(1)

cols = list(df.columns)
print("FILE:", p)
print("COLUMNS:", cols)
print("DTYPES:", {c: str(df[c].dtype) for c in cols[:12]})
for idxcol in ("__index_level_0__",):
    if idxcol in df.columns:
        print(f"NOTE: found special index column: {idxcol}; dtype={df[idxcol].dtype}")
print("\nHEAD:")
print(df.head(3).to_string(index=False))

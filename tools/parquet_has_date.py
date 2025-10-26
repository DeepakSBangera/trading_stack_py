import sys

import pandas as pd

if len(sys.argv) < 2:
    print("NO path_missing")
    sys.exit(2)

fn = sys.argv[1]
try:
    df = pd.read_parquet(fn)
except Exception as e:
    print("NO read_error", str(e).replace("\n", " ")[:200])
    sys.exit(1)

cols = [c.lower() for c in df.columns]
if "date" in cols:
    print("OK")
    sys.exit(0)
else:
    print("NO no_date_col")
    sys.exit(3)

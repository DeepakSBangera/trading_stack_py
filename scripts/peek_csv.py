import json
import sys
from pathlib import Path

import pandas as pd

p = Path(sys.argv[1]) if len(sys.argv) > 1 else None
if not p or not p.exists():
    print("Usage: python scripts/peek_csv.py <csv>")
    raise SystemExit(1)
df = pd.read_csv(p)
print(json.dumps({"path": str(p), "rows": int(df.shape[0]), "cols": list(df.columns)}, indent=2))
print(df.head(10).to_string(index=False))

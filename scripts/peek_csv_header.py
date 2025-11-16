# 2) FULL FILE CONTENTS — paste all below
from __future__ import annotations

from pathlib import Path

import pandas as pd

p = Path(r"F:\Projects\trading_stack_py\reports\wk12_orders_lastday.csv")
if not p.exists():
    raise SystemExit(f"Missing file: {p}")
df = pd.read_csv(p, nrows=10)
print("Columns:", list(df.columns))
print(df.head(5).to_string(index=False))

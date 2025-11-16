import sys

import numpy as np
import pandas as pd
from trading_stack_py.cv.walkforward import WalkForwardCV

csv, date_col, price_col, train, test, step, embargo, expanding, min_train = sys.argv[1:]
train, test, step, embargo = map(int, (train, test, step, embargo))
expanding = expanding.lower() == "true"
min_train = None if min_train == "none" else int(min_train)

df = pd.read_csv(csv)
p = pd.to_numeric(df[price_col], errors="coerce").ffill()
r = p.pct_change().to_numpy()
r = r[~np.isnan(r)]
n = len(r)

cv = WalkForwardCV(
    train_size=train,
    test_size=test,
    step_size=step,
    expanding=expanding,
    embargo=embargo,
    min_train_size=min_train,
)
splits = list(cv.split(n))
print(f"n_returns={n}, segments={len(splits)}")
if splits:
    for i, (tr, te) in enumerate(splits[:3], 1):
        print(f"seg{i}: train[{tr[0]}..{tr[-1]}] len={len(tr)}, test[{te[0]}..{te[-1]}] len={len(te)}")

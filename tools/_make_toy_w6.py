from pathlib import Path

import numpy as np

root = Path(r"F:\Projects\trading_stack_py\reports\W6")
(root).mkdir(parents=True, exist_ok=True)

for seg in (1, 2):
    seg_dir = root / f"segment_{seg:02d}"
    seg_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(40 + seg)
    n_tr, n_te, d = 80, 30, 8

    Xtr = rng.normal(size=(n_tr, d)).astype("float32")
    coef = rng.normal(size=(d,))
    ytr = (Xtr @ coef + rng.normal(scale=0.5, size=n_tr)).astype("float32")

    Xte = rng.normal(size=(n_te, d)).astype("float32")
    yte = (Xte @ coef + rng.normal(scale=0.5, size=n_te)).astype("float32")

    np.save(seg_dir / "X_train.npy", Xtr)
    np.save(seg_dir / "y_train.npy", ytr)
    np.save(seg_dir / "X_test.npy", Xte)
    np.save(seg_dir / "y_test.npy", yte)

print("W6 toy data created at:", root)

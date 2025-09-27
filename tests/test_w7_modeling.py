import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from trading_stack_py.pipelines.train_models import (
    load_segments,
    train_classification_per_segment,
    train_regression_per_segment,
)


def _make_toy_w6(tmp: Path):
    # create segments.csv and two segments with minimal arrays
    (tmp / "segment_01").mkdir(parents=True, exist_ok=True)
    (tmp / "segment_02").mkdir(parents=True, exist_ok=True)
    segs = pd.DataFrame([{"segment": 1}, {"segment": 2}])
    segs.to_csv(tmp / "segments.csv", index=False)
    # segment 1
    np.save(tmp / "segment_01" / "X_train.npy", np.random.randn(40, 3))
    np.save(tmp / "segment_01" / "y_train.npy", np.random.randn(40))
    np.save(tmp / "segment_01" / "X_test.npy", np.random.randn(10, 3))
    np.save(tmp / "segment_01" / "y_test.npy", np.random.randn(10))
    # segment 2 (classification-ish)
    Xtr = np.random.randn(50, 3)
    ytr = (np.random.randn(50) > 0).astype(float)
    Xte = np.random.randn(12, 3)
    yte = (np.random.randn(12) > 0).astype(float)
    np.save(tmp / "segment_02" / "X_train.npy", Xtr)
    np.save(tmp / "segment_02" / "y_train.npy", ytr)
    np.save(tmp / "segment_02" / "X_test.npy", Xte)
    np.save(tmp / "segment_02" / "y_test.npy", yte)


def test_trainers_run():
    tmp = Path(tempfile.mkdtemp())
    try:
        _make_toy_w6(tmp)
        recs = load_segments(tmp)
        assert len(recs) == 2
        reg = train_regression_per_segment([recs[0]])
        clf = train_classification_per_segment([recs[1]])
        assert not reg.empty and not clf.empty
    finally:
        shutil.rmtree(tmp)

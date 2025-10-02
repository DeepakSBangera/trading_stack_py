from pathlib import Path

import numpy as np
import pandas as pd

from trading_stack_py.pipelines.tune_models import run_tuning


def _make_toy_w6(tmp: Path) -> Path:
    tmp.mkdir(parents=True, exist_ok=True)
    # Two segments
    (tmp / "segment_01").mkdir(parents=True, exist_ok=True)
    (tmp / "segment_02").mkdir(parents=True, exist_ok=True)
    segs = pd.DataFrame([{"segment": 1}, {"segment": 2}])
    segs.to_csv(tmp / "segments.csv", index=False)

    # segment 1 (regression-y)
    rng = np.random.default_rng(0)
    np.save(tmp / "segment_01" / "X_train.npy", rng.normal(size=(40, 4)))
    np.save(tmp / "segment_01" / "y_train.npy", rng.normal(size=(40,)))
    np.save(tmp / "segment_01" / "X_test.npy", rng.normal(size=(10, 4)))
    np.save(tmp / "segment_01" / "y_test.npy", rng.normal(size=(10,)))

    # segment 2 (also regression-y)
    np.save(tmp / "segment_02" / "X_train.npy", rng.normal(size=(50, 4)))
    np.save(tmp / "segment_02" / "y_train.npy", rng.normal(size=(50,)))
    np.save(tmp / "segment_02" / "X_test.npy", rng.normal(size=(12, 4)))
    np.save(tmp / "segment_02" / "y_test.npy", rng.normal(size=(12,)))
    return tmp


def test_tune_writes_outputs(tmp_path: Path):
    w6 = _make_toy_w6(tmp_path / "W6")
    out_root = run_tuning(w6, task="regression", tag="TEST_W8", outdir=tmp_path / "W8")

    best = out_root / "best_params.csv"
    readme = out_root / "README.md"
    assert best.exists() and readme.exists()

    df = pd.read_csv(best)
    # required columns
    for col in ["segment", "alpha", "mse", "r2", "pnl_proxy", "n_test"]:
        assert col in df.columns
    # two segments tuned
    assert len(df) == 2

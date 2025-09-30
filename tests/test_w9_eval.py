from pathlib import Path

import numpy as np
import pandas as pd

from trading_stack_py.pipelines.evaluate_models import evaluate


def _make_toy_w6(tmp: Path) -> Path:
    # Minimal segments.csv with two segments
    segs = pd.DataFrame(
        [
            {
                "segment": 1,
                "train_start": 0,
                "train_end": 10,
                "test_start": 10,
                "test_end": 12,
                "train_len": 11,
                "test_len": 2,
            },
            {
                "segment": 2,
                "train_start": 2,
                "train_end": 12,
                "test_start": 12,
                "test_end": 14,
                "train_len": 11,
                "test_len": 2,
            },
        ]
    )
    (tmp / "segment_01").mkdir(parents=True, exist_ok=True)
    (tmp / "segment_02").mkdir(parents=True, exist_ok=True)
    # Arrays are not required by evaluator, but match real layout
    np.save(tmp / "segment_01" / "X_train.npy", np.random.randn(3, 2))
    np.save(tmp / "segment_01" / "y_train.npy", np.random.randn(3))
    np.save(tmp / "segment_01" / "X_test.npy", np.random.randn(1, 2))
    np.save(tmp / "segment_01" / "y_test.npy", np.random.randn(1))
    np.save(tmp / "segment_02" / "X_train.npy", np.random.randn(3, 2))
    np.save(tmp / "segment_02" / "y_train.npy", np.random.randn(3))
    np.save(tmp / "segment_02" / "X_test.npy", np.random.randn(1, 2))
    np.save(tmp / "segment_02" / "y_test.npy", np.random.randn(1))
    segs.to_csv(tmp / "segments.csv", index=False)
    return tmp


def _make_toy_w7(tmp: Path) -> Path:
    # Minimal per-segment metrics (regression flavor)
    mets = pd.DataFrame(
        [
            {"segment": 1, "mse": 0.1, "r2": 0.0, "pnl_proxy": 1.0, "n_test": 2},
            {"segment": 2, "mse": 0.2, "r2": -0.1, "pnl_proxy": -0.5, "n_test": 2},
        ]
    )
    mets.to_csv(tmp / "segment_metrics.csv", index=False)
    return tmp


def test_w9_evaluator_writes_joined_and_readme(tmp_path: Path):
    w6 = _make_toy_w6(tmp_path / "W6")
    w7 = _make_toy_w7(tmp_path / "W7")
    out = tmp_path / "W9"

    root = evaluate(w6, w7, out, tag="TEST_W9")
    joined = pd.read_csv(root / "joined.csv")

    # Basic assertions
    assert (root / "README.md").exists()
    assert len(joined) == 2
    # Must contain keys from both sources
    for col in ("segment", "train_start", "test_end", "mse", "pnl_proxy"):
        assert col in joined.columns

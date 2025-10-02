from __future__ import annotations

from pathlib import Path

import pandas as pd

from trading_stack_py.pipelines.evaluate_models import evaluate


def _make_toy_w6(tmp: Path) -> Path:
    tmp.mkdir(parents=True, exist_ok=True)
    segs = pd.DataFrame({"segment": [1, 2, 3], "dummy": [0, 0, 0]})
    segs.to_csv(tmp / "segments.csv", index=False)
    return tmp


def _make_toy_w7(tmp: Path) -> Path:
    tmp.mkdir(parents=True, exist_ok=True)
    mets = pd.DataFrame(
        {
            "segment": [1, 2, 3],
            "mse": [0.1, 0.2, 0.3],
            "r2": [0.01, -0.02, 0.0],
            "pnl_proxy": [1.0, -0.5, 0.2],
        }
    )
    mets.to_csv(tmp / "segment_metrics.csv", index=False)
    return tmp


def test_w9_evaluator_writes_joined_and_readme(tmp_path: Path):
    w6 = _make_toy_w6(tmp_path / "W6")
    w7 = _make_toy_w7(tmp_path / "W7")
    out = tmp_path / "W9"

    root = evaluate(w6, w7, out_root=out, tag="TEST_W9")
    assert isinstance(root, Path)
    assert root.exists()
    assert (root / "joined.csv").exists()
    assert (root / "README.md").exists()

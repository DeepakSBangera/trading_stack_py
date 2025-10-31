import numpy as np

from trading_stack_py.metrics.dsr import (
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
)


def test_psr_monotonicity():
    # With fixed n, skew, kurt, a larger SR should yield higher PSR
    psr1 = probabilistic_sharpe_ratio(0.2, 0.0, 252, 0.0, 3.0)
    psr2 = probabilistic_sharpe_ratio(0.5, 0.0, 252, 0.0, 3.0)
    assert psr2 > psr1 >= 0 and psr2 <= 1


def test_dsr_bounds():
    rng = np.random.default_rng(0)
    sr_samples = rng.normal(0.2, 0.1, size=20)
    dsr = deflated_sharpe_ratio(sr_samples, num_trials=20)
    assert 0.0 <= dsr <= 1.0

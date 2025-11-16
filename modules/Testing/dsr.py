from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis, norm, skew

from . import _dsr_internal_helpers as _h  # created below


def probabilistic_sharpe_ratio(sr_hat: float, sr_threshold: float, n: int, skewness: float, kurt: float) -> float:
    """
    PSR per López de Prado & Bailey.
    PSR = Phi( ((sr_hat - sr_threshold)*sqrt(n-1)) / sqrt(1 - skewness*sr_hat + ((kurt - 1)/4.0)*sr_hat**2) )
    """
    if n <= 1:
        return 0.0
    num = (sr_hat - sr_threshold) * np.sqrt(n - 1.0)
    den = np.sqrt(1.0 - skewness * sr_hat + ((kurt - 1.0) / 4.0) * (sr_hat**2))
    z = num / den if den > 0 else -np.inf
    return float(norm.cdf(z))


def expected_noise_sr0(num_trials: int, sr_cross_var: float) -> float:
    """
    Expected maximum Sharpe under N unskilled trials (threshold SR_0).
    SR_0 = sqrt(Var_hat(SR)) * [ (1-γ) Φ^{-1}(1 - 1/N) + γ Φ^{-1}(1 - 1/(N e)) ].
    """
    if num_trials <= 1:
        return 0.0
    return _h.sr0_from_var_and_trials(sr_cross_var, num_trials)


def deflated_sharpe_ratio(returns: np.ndarray, num_trials: int, sr_cross_var: float | None = None) -> float:
    """
    DSR = PSR with threshold SR_0 (expected max SR from noise).
    Skew & kurtosis are computed from sample returns.
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    n = r.size
    if n <= 1:
        return 0.0
    sr_hat = r.mean() / r.std(ddof=1)
    g3 = float(skew(r, bias=False))
    g4 = float(kurtosis(r, fisher=False, bias=False))  # Pearson (normal => 3.0)
    # If cross-sectional SR variance unknown, use asymptotic variance of SR estimator as proxy
    if sr_cross_var is None:
        sr_cross_var = _h.asymptotic_sr_variance(sr_hat, n, g3, g4)
    sr0 = expected_noise_sr0(num_trials, sr_cross_var)
    return probabilistic_sharpe_ratio(sr_hat, sr0, n, g3, g4)

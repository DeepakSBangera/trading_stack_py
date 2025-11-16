from __future__ import annotations

import numpy as np
from scipy.stats import norm


def minimum_track_record_length(
    sr_hat: float,
    p_star: float = 0.95,
    sr_threshold: float = 0.0,
    skewness: float = 0.0,
    kurt: float = 3.0,
) -> float:
    """
    Solve for smallest n such that PSR(sr_hat; n, skew, kurt, sr_threshold) >= p_star.
    n ≈ 1 + [ Φ^{-1}(p*) * sqrt(1 - skew*SR + ((kurt-1)/4) SR^2) / (SR - SR_thr) ]^2
    Returns a float (you can ceil() for integer days).
    """
    sr = float(sr_hat)
    if sr <= sr_threshold:
        return np.inf
    numer = norm.ppf(p_star) * np.sqrt(1.0 - skewness * sr + ((kurt - 1.0) / 4.0) * (sr**2))
    denom = sr - sr_threshold
    n = 1.0 + (numer / denom) ** 2
    return float(n)

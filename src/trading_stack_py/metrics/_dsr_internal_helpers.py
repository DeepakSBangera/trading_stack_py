from __future__ import annotations

import numpy as np
from scipy.stats import norm

EULER_MASCHERONI = 0.5772156649015329


def sr0_from_var_and_trials(sr_var: float, N: int) -> float:
    if N <= 1 or sr_var <= 0:
        return 0.0
    q1 = norm.ppf(1.0 - 1.0 / float(N))
    q2 = norm.ppf(1.0 - 1.0 / (float(N) * np.e))
    return float(np.sqrt(sr_var) * ((1.0 - EULER_MASCHERONI) * q1 + EULER_MASCHERONI * q2))


def asymptotic_sr_variance(sr_hat: float, n: int, g3: float, g4: float) -> float:
    # Var(SR) ≈ (1 + 0.5*SR^2 - g3*SR + (g4-3)/4 * SR^2) / (n - 1)
    if n <= 1:
        return np.nan
    num = 1.0 + 0.5 * (sr_hat**2) - g3 * sr_hat + ((g4 - 3.0) / 4.0) * (sr_hat**2)
    return float(num / (n - 1.0))

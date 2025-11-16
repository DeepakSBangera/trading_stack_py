from __future__ import annotations

# ---- factors
from .factors.exposures import (
    load_sector_mapping as load_sector_mapping,
)
from .factors.exposures import (
    momentum_12_1_proxy as momentum_12_1_proxy,
)
from .factors.exposures import (
    quality_inverse_downside_vol as quality_inverse_downside_vol,
)
from .factors.exposures import (
    rolling_sector_exposures_from_weights as rolling_sector_exposures_from_weights,
)
from .metrics.attribution import (
    align_weights_and_returns as align_weights_and_returns,
)
from .metrics.attribution import (
    build_returns_from_prices as build_returns_from_prices,
)
from .metrics.attribution import (
    contribution_by_ticker as contribution_by_ticker,
)
from .metrics.attribution import (
    group_contribution as group_contribution,
)
from .metrics.attribution import (
    pivot_weights as pivot_weights,
)
from .metrics.sharpe import (
    sharpe_annual as sharpe_annual,
)

# Public package surface (clean, minimal)
# ---- metrics
from .metrics.sharpe import (
    sharpe_daily as sharpe_daily,
)

# Backward-compat aliases expected elsewhere
momentum_proxy_12_1_from_nav = momentum_12_1_proxy
quality_proxy_inv_downside_vol = quality_inverse_downside_vol

__all__ = [
    # metrics
    "sharpe_daily",
    "sharpe_annual",
    "align_weights_and_returns",
    "build_returns_from_prices",
    "contribution_by_ticker",
    "group_contribution",
    "pivot_weights",
    # factors
    "load_sector_mapping",
    "rolling_sector_exposures_from_weights",
    "momentum_12_1_proxy",
    "quality_inverse_downside_vol",
    # aliases
    "momentum_proxy_12_1_from_nav",
    "quality_proxy_inv_downside_vol",
]

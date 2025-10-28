"""
Public metrics API surface.

Only re-export stable entry points we intend users/tools to import.
"""

# Attribution helpers (these are part of the metrics namespace in this project)
from .attribution import (
    align_weights_and_returns as align_weights_and_returns,
)
from .attribution import (
    build_returns_from_prices as build_returns_from_prices,
)
from .attribution import (
    contribution_by_ticker as contribution_by_ticker,
)
from .attribution import (
    group_contribution as group_contribution,
)
from .attribution import (
    pivot_weights as pivot_weights,
)
from .calmar import calmar_ratio as calmar_ratio
from .drawdown import max_drawdown as max_drawdown
from .omega import omega_ratio as omega_ratio
from .sharpe import sharpe_annual as sharpe_annual
from .sortino import sortino_annual as sortino_annual

__all__ = [
    # point metrics
    "calmar_ratio",
    "max_drawdown",
    "omega_ratio",
    "sharpe_annual",
    "sortino_annual",
    # attribution helpers
    "align_weights_and_returns",
    "build_returns_from_prices",
    "contribution_by_ticker",
    "group_contribution",
    "pivot_weights",
]

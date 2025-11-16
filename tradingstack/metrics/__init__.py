from __future__ import annotations

"""
Public metrics API surface.
Only re-export stable entry points we intend users/tools to import.
"""

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
from .sharpe import sharpe_annual as sharpe_annual
from .sharpe import sharpe_daily as sharpe_daily

__all__ = [
    "sharpe_daily",
    "sharpe_annual",
    "align_weights_and_returns",
    "build_returns_from_prices",
    "contribution_by_ticker",
    "group_contribution",
    "pivot_weights",
]

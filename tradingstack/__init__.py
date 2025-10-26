# Central exports, stable public API via compat shim (no breaking renames)

# Optional IO exports (keep working even if IO package not present yet)
try:
    from tradingstack.io.equity import *  # noqa: F401,F403
except Exception:
    pass

# Point metrics via compatibility layer (maps to your actual function names)
from tradingstack.metrics.compat import (
    calmar_ratio,
    max_drawdown,
    omega_ratio,
    sharpe_ratio,
    sortino_ratio,
)

# Rolling metrics (Session 2)
from tradingstack.metrics.rolling import (
    compute_rolling_metrics_from_nav,
    rolling_drawdown,
    rolling_sharpe,
    rolling_sortino,
    rolling_volatility,
    trend_regime,
)

# Optional attribution exports, if present
try:
    from tradingstack.metrics.attribution import *  # noqa: F401,F403
except Exception:
    pass

__all__ = [
    # point metrics (normalized)
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "calmar_ratio",
    "omega_ratio",
    # rolling metrics
    "rolling_volatility",
    "rolling_sharpe",
    "rolling_sortino",
    "rolling_drawdown",
    "trend_regime",
    "compute_rolling_metrics_from_nav",
]

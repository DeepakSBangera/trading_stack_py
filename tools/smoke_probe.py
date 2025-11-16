from __future__ import annotations

from tradingstack.factors import momentum_12_1_proxy, quality_inverse_downside_vol
from tradingstack.metrics import sharpe_annual, sharpe_daily

print(
    "imports ok:",
    callable(sharpe_daily),
    callable(sharpe_annual),
    callable(momentum_12_1_proxy),
    callable(quality_inverse_downside_vol),
)

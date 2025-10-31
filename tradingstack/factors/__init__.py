from __future__ import annotations

from .exposures import (
    FactorOutputs,
    load_sector_mapping,
    momentum_12_1_proxy,
    quality_inverse_downside_vol,
    rolling_sector_exposures_from_weights,
)

# Backward-compatible aliases (old names some scripts expect)
momentum_proxy_12_1_from_nav = momentum_12_1_proxy
quality_proxy_inv_downside_vol = quality_inverse_downside_vol

__all__ = [
    "FactorOutputs",
    "load_sector_mapping",
    "rolling_sector_exposures_from_weights",
    "momentum_12_1_proxy",
    "quality_inverse_downside_vol",
    "momentum_proxy_12_1_from_nav",
    "quality_proxy_inv_downside_vol",
]

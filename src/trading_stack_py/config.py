# src/trading_stack_py/config.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# Defaults if config/strategy.yml is missing or partial
_DEFAULTS = {
    "signals": {
        "ma_fast": 20,
        "ma_slow": 200,
        "use_crossover": False,  # default = Close > fast; set True for fast>slow
    },
    "portfolio": {
        "top_n": 5,  # breadth for portfolio mode (not used by single-ticker run)
        "rebal_freq": "ME",  # month-end
        "cost_bps": 10,  # round-trip bps used by CLI if not overridden
        "cash_buffer": 0.00,
    },
}


@dataclass
class StrategyConfig:
    ma_fast: int
    ma_slow: int
    use_crossover: bool
    portfolio: dict[str, Any]


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_strategy_config(config_dir: Path | None = None) -> StrategyConfig:
    """
    Loads config from config/strategy.yml or config/strategy.yaml.
    Falls back to sane defaults if not found or keys are missing.
    """
    base = Path(config_dir) if config_dir else Path("config")
    yml = base / "strategy.yml"
    yaml_ = base / "strategy.yaml"

    user_cfg = _load_yaml(yml if yml.exists() else yaml_) if base.exists() else {}

    # Merge shallowly against defaults
    merged = {
        "signals": {**_DEFAULTS["signals"], **(user_cfg.get("signals") or {})},
        "portfolio": {**_DEFAULTS["portfolio"], **(user_cfg.get("portfolio") or {})},
    }

    return StrategyConfig(
        ma_fast=int(merged["signals"]["ma_fast"]),
        ma_slow=int(merged["signals"]["ma_slow"]),
        use_crossover=bool(merged["signals"]["use_crossover"]),
        portfolio=merged["portfolio"],
    )


def get_portfolio_params(config_dir: Path | None = None) -> dict[str, Any]:
    """
    Convenience: returns the 'portfolio' dict (e.g., to read cost_bps).
    Kept for backwards compatibility with earlier imports.
    """
    cfg = load_strategy_config(config_dir=config_dir)
    return dict(cfg.portfolio)

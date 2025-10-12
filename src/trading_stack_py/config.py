from __future__ import annotations

import functools
import os
from typing import Any

try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # optional dependency


@functools.lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    """Loads config/strategy.yaml if present; returns {} otherwise."""
    path = os.path.join("config", "strategy.yaml")
    if yaml is None or not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    # light sanity defaults
    return {
        "top_n": data.get("top_n", 5),
        "rebal_freq": data.get("rebal_freq", "ME"),
        "cost_bps": data.get("cost_bps", 10),
        "cash_buffer": data.get("cash_buffer", 0.00),
        "moving_average": {
            "fast": data.get("moving_average", {}).get("fast", 20),
            "slow": data.get("moving_average", {}).get("slow", 200),
        },
    }


def get_default_ma_params() -> tuple[int, int]:
    cfg = load_config()
    ma = cfg.get("moving_average", {})
    return int(ma.get("fast", 20)), int(ma.get("slow", 200))


def get_portfolio_params() -> dict[str, Any]:
    """Convenience accessor for portfolio-level knobs."""
    c = load_config()
    return {
        "top_n": int(c.get("top_n", 5)),
        "rebal_freq": str(c.get("rebal_freq", "ME")),
        "cost_bps": float(c.get("cost_bps", 10)),
        "cash_buffer": float(c.get("cash_buffer", 0.00)),
    }

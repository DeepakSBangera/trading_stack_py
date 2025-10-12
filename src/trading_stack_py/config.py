# src/trading_stack_py/config.py
from __future__ import annotations

from pathlib import Path

import yaml

_DEFAULTS = {
    "portfolio": {"cost_bps": 10, "top_n": 5, "rebal_freq": "ME", "cash_buffer": 0.0},
    "signals": {"ma": {"fast": 20, "slow": 200, "use_crossover": False}},
}


def _load_yaml() -> dict:
    # support both strategy.yaml and strategy.yml
    for name in ("config/strategy.yaml", "config/strategy.yml"):
        p = Path(name)
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    return {}


def get_portfolio_params() -> dict:
    data = _load_yaml()
    return {**_DEFAULTS["portfolio"], **(data.get("portfolio") or {})}


def get_default_ma_params() -> tuple[int, int]:
    data = _load_yaml()
    ma = (data.get("signals") or {}).get("ma") or {}
    fast = ma.get("fast", _DEFAULTS["signals"]["ma"]["fast"])
    slow = ma.get("slow", _DEFAULTS["signals"]["ma"]["slow"])
    try:
        fast = int(fast)
    except Exception:
        fast = _DEFAULTS["signals"]["ma"]["fast"]
    try:
        slow = int(slow)
    except Exception:
        slow = _DEFAULTS["signals"]["ma"]["slow"]
    return fast, slow


def get_signal_flags() -> dict:
    data = _load_yaml()
    ma = (data.get("signals") or {}).get("ma") or {}
    use_crossover = bool(ma.get("use_crossover", _DEFAULTS["signals"]["ma"]["use_crossover"]))
    return {"use_crossover": use_crossover}

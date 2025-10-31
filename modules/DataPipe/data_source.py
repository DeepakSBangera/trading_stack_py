import json
import os
from dataclasses import dataclass

CFG_PATH = os.path.join("config", "data_source.json")


@dataclass
class SynthCfg:
    prices_root: str
    fundamentals_root: str


@dataclass
class KiteCfg:
    enabled: bool
    dry_run: bool
    cache_root: str
    rate_limit_per_min: int
    backoff_seconds: float
    universe_file: str
    pit_log: str


@dataclass
class DataSource:
    mode: str
    synth: SynthCfg
    kite: KiteCfg


def _utf8_sig_tolerant_read(path: str) -> str:
    # tolerate BOM if any
    with open(path, "rb") as f:
        raw = f.read()
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("utf-8")


def load_config(path: str = CFG_PATH) -> DataSource:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    cfg = json.loads(_utf8_sig_tolerant_read(path))
    synth = cfg.get("synth", {})
    kite = cfg.get("kite", {})
    return DataSource(
        mode=cfg.get("mode", "synth"),
        synth=SynthCfg(
            prices_root=synth.get("prices_root", "data_synth/prices"),
            fundamentals_root=synth.get("fundamentals_root", "data_synth/fundamentals"),
        ),
        kite=KiteCfg(
            enabled=bool(kite.get("enabled", False)),
            dry_run=bool(kite.get("dry_run", True)),
            cache_root=kite.get("cache_root", "data_live"),
            rate_limit_per_min=int(kite.get("rate_limit_per_min", 90)),
            backoff_seconds=float(kite.get("backoff_seconds", 2.0)),
            universe_file=kite.get("universe_file", "config/universe_kite.csv"),
            pit_log=kite.get("pit_log", "data_live/_pit/log.jsonl"),
        ),
    )


def current_mode() -> str:
    return load_config().mode


def kite_enabled() -> bool:
    ds = load_config()
    return ds.mode == "kite" and ds.kite.enabled


def synth_roots():
    ds = load_config()
    return ds.synth.prices_root, ds.synth.fundamentals_root


def kite_paths():
    ds = load_config()
    return ds.kite.cache_root, ds.kite.universe_file, ds.kite.pit_log

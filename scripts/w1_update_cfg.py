import yaml

with open("config/config.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

cfg.setdefault("data", {})
cfg["data"]["source"] = "csv"
cfg["data"]["csv_dir"] = cfg["data"].get("csv_dir", "data/csv")

cfg.setdefault("signals", {})
cfg["signals"]["rule"] = "R2_momo_rsi"
p = cfg["signals"].get("params", {}) or {}
p.update(
    {
        "rsi_buy": 52,  # lenient so we see some buys on seeded data
        "atr_min_pct": 0.005,  # 0.5%–10% ATR band
        "atr_max_pct": 0.10,
    }
)
cfg["signals"]["params"] = p

with open("config/config.yaml", "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

print("Updated config/config.yaml for W1 (source=csv, rule=R2_momo_rsi)")

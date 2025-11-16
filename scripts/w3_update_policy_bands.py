from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(r"F:\Projects\trading_stack_py")
REPORTS = ROOT / "reports"
CONFIG = ROOT / "config" / "capacity_policy.yaml"
BACKUP = ROOT / "config" / "capacity_policy.backup.yaml"


def open_win(p: Path):
    if sys.platform.startswith("win") and p.exists():
        os.startfile(p)


def load_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def dump_yaml(path: Path, data: dict):
    import yaml

    text = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    path.write_text(text, encoding="utf-8")


def main():
    rec_csv = REPORTS / "band_recommendations.csv"
    if not rec_csv.exists():
        raise SystemExit(f"Missing recommendations CSV: {rec_csv} — run w3_band_utilization.py first.")
    if not CONFIG.exists():
        raise SystemExit(f"Missing policy: {CONFIG}")

    rec_df = pd.read_csv(rec_csv)
    # Build dict like {"L1": 0.9, "L2": 1.3, ...}
    rec_map = {row["list_tier"]: float(row["recommended_band_pct"]) for _, row in rec_df.iterrows()}

    pol = load_yaml(CONFIG)
    if "turnover_bands_pct_per_day" not in pol:
        pol["turnover_bands_pct_per_day"] = {}
    bands = pol["turnover_bands_pct_per_day"]

    # Backup existing YAML
    BACKUP.write_text(CONFIG.read_text(encoding="utf-8"), encoding="utf-8")

    # Apply recommendations only for tiers present in rec_map
    for k, v in rec_map.items():
        bands[str(k)] = float(v)

    # Save and reopen
    dump_yaml(CONFIG, pol)

    print(
        json.dumps(
            {"backup": str(BACKUP), "updated_config": str(CONFIG), "new_bands": bands},
            indent=2,
        )
    )

    open_win(CONFIG)  # open the YAML so you can see the new bands


if __name__ == "__main__":
    main()

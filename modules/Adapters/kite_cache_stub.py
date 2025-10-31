import datetime as dt
import json
import os
import sys
from pathlib import Path

from data_source import kite_enabled, kite_paths, load_config


def ensure_dirs(*paths):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def append_pit(log_path: str, event: dict):
    Path(os.path.dirname(log_path)).mkdir(parents=True, exist_ok=True)
    event["ts"] = dt.datetime.utcnow().isoformat() + "Z"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def main():
    ds = load_config()
    cache_root, universe_csv, pit_log = kite_paths()
    ensure_dirs(
        cache_root, os.path.join(cache_root, "_pit"), os.path.dirname(universe_csv)
    )

    # Create a placeholder universe file if missing
    if not os.path.exists(universe_csv):
        with open(universe_csv, "w", encoding="utf-8") as f:
            f.write("ticker,tradingsymbol,exchange\n")  # you can fill this later

    if not kite_enabled():
        append_pit(
            pit_log, {"mode": ds.mode, "kite.enabled": False, "action": "scaffold_only"}
        )
        print("Kite scaffold ready (enabled=False). No API calls.")
        print(f"cache_root={cache_root}")
        print(f"universe_file={universe_csv}")
        print(f"pit_log={pit_log}")
        return

    # Even if enabled=True, we DO NOT call the API yet. Just log intent.
    append_pit(
        pit_log,
        {
            "mode": ds.mode,
            "kite.enabled": True,
            "dry_run": ds.kite.dry_run,
            "action": "dry_run_only",
            "note": "no API calls in this stub",
        },
    )
    print("Kite enabled=True but this is a dry-run stub. No API calls made.")
    print(f"cache_root={cache_root}")
    print(f"universe_file={universe_csv}")
    print(f"pit_log={pit_log}")


if __name__ == "__main__":
    sys.exit(main())

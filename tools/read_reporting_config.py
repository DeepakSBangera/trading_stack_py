from __future__ import annotations

import json
from pathlib import Path

# Python 3.11: tomllib is stdlib; fallback to tomli if needed
try:
    import tomllib  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

DEFAULTS = {
    "start": "2025-01-01",
    "universe_csv": "config/universe.csv",
    "rolling_window": 63,
    "open_after": True,
    "archive_after": False,
}


def main() -> None:
    cfg_path = Path("config/reporting.toml")
    data = dict(DEFAULTS)
    if cfg_path.exists():
        with cfg_path.open("rb") as f:
            try:
                parsed = tomllib.load(f) or {}
            except Exception:
                parsed = {}
        for k, v in parsed.items():
            if k in DEFAULTS:
                data[k] = v
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()

# tools/schema_guard.py
import glob
import json
import os
import re
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

REQUIRED = {
    "prices": {
        "date": ("timestamp", "ANY_TZ_OR_NAIVE"),
        "symbol": "string",
        "close": "float",  # accept any floating type
    },
    "fundamentals": {
        "date": ("timestamp", "ANY_TZ_OR_NAIVE"),
        "symbol": "string",
        "metric": "string",
        "value": "float",
    },
    "equity": {
        "date": ("timestamp", "ANY_TZ_OR_NAIVE"),
        ("value", "equity"): "float",
    },
}

SUFFIX_SKIP = re.compile(r"_(weights|trades|tearsheet)\.parquet$", re.IGNORECASE)


def _is_ts(dt: pa.DataType) -> bool:
    return pa.types.is_timestamp(dt)


def _is_float(dt: pa.DataType) -> bool:
    return pa.types.is_floating(dt)


def _type_ok(col_type: pa.DataType, spec) -> bool:
    if spec == "float":
        return _is_float(col_type)
    if isinstance(spec, tuple) and spec[0] == "timestamp":
        return _is_ts(col_type)
    if spec == "string":
        return pa.types.is_string(col_type) or pa.types.is_large_string(col_type)
    # fallback exact match string
    return str(col_type) == spec


def _expand(arg: str):
    p = Path(arg)
    if p.is_dir():
        return [str(x) for x in p.rglob("*.parquet")]
    hits = glob.glob(arg)
    if hits:
        return [str(Path(h)) for h in hits]
    if p.exists():
        return [str(p)]
    return []


def check_file(kind: str, path: str):
    sch = pq.read_schema(path)
    req = REQUIRED[kind]

    names = list(sch.names)
    types = {n: sch.field(n).type for n in names}

    missing, wrong, warn = [], [], []

    # warn if timestamp exists but not tz-aware
    def _tz_note(nm):
        t = types.get(nm)
        if t and pa.types.is_timestamp(t) and (t.tz is None):
            warn.append(f"column '{nm}' is naive timestamp (treat as UTC).")

    for key, spec in req.items():
        if isinstance(key, tuple):
            present = next((k for k in key if k in names), None)
            if present is None:
                missing.append(f"one of {key}")
            else:
                if not _type_ok(types[present], spec):
                    wrong.append((present, str(types[present]), spec))
                if isinstance(spec, tuple) and spec[0] == "timestamp":
                    _tz_note(present)
        else:
            if key not in names:
                missing.append(key)
            else:
                if not _type_ok(types[key], spec):
                    wrong.append((key, str(types[key]), spec))
                if isinstance(spec, tuple) and spec[0] == "timestamp":
                    _tz_note(key)

    return {"path": path, "missing": missing, "wrong": wrong, "warn": warn}


def main():
    if len(sys.argv) < 3:
        print(
            "Usage: schema_guard.py <prices|fundamentals|equity> <file|dir|glob> [more...]"
        )
        sys.exit(1)

    kind = sys.argv[1].lower()
    if kind not in REQUIRED:
        print(f"Unknown kind '{kind}'. Choose: {', '.join(REQUIRED.keys())}")
        sys.exit(1)

    raw = sys.argv[2:]
    files = []
    for a in raw:
        files += _expand(a)
    files = sorted(set(files))

    # auto-filter non-equity artifacts for kind=equity
    if kind == "equity":
        files = [f for f in files if not SUFFIX_SKIP.search(os.path.basename(f))]

    if not files:
        print("No files matched.")
        sys.exit(1)

    results, any_fail = [], False
    for f in files:
        try:
            res = check_file(kind, f)
            results.append(res)
            if res["missing"] or res["wrong"]:
                any_fail = True
        except Exception as e:
            any_fail = True
            results.append(
                {
                    "path": f,
                    "error": str(e),
                    "missing": ["<unknown>"],
                    "wrong": [],
                    "warn": [],
                }
            )

    print(json.dumps(results, indent=2))
    sys.exit(2 if any_fail else 0)


if __name__ == "__main__":
    main()

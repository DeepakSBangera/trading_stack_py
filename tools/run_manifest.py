import argparse
import glob
import hashlib
import json
import os
from pathlib import Path


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--week", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--reports", required=True)
    ap.add_argument("--git-sha", required=True)
    ap.add_argument("--when", required=True)
    args = ap.parse_args()

    out = {
        "week": args.week,
        "git_sha": args.git_sha,
        "when": args.when,
        "config": args.config,
        "config_sha": sha(Path(args.config)),
        "reports_dir": args.reports,
        "reports_sample": [],
    }
    try:
        recent = sorted(glob.glob(os.path.join(args.reports, "*.*")), key=os.path.getmtime)[-10:]
        for p in recent:
            out["reports_sample"].append({"path": p, "bytes": os.path.getsize(p)})
    except Exception:
        pass

    manifest = Path(args.reports) / "run_manifest.jsonl"
    with open(manifest, "a", encoding="utf-8") as f:
        f.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()

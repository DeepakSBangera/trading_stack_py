import argparse
import glob
import hashlib
import json
import os
import platform
from datetime import datetime

import pandas as pd


def latest(pat):
    files = sorted(glob.glob(pat), key=os.path.getmtime, reverse=True)
    return files[0] if files else None


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def pick_payloads(reports):
    picks = {}
    for kind in ("", "_weights", "_trades"):
        pq = latest(os.path.join(reports, f"portfolioV2*{kind}.parquet"))
        if pq:
            picks[kind or "equity"] = pq
    # tearsheet parquet (optional)
    ts = latest(os.path.join(reports, "portfolioV2*_*tearsheet.parquet"))
    if ts:
        picks["tearsheet"] = ts
    return picks


def try_tearsheet_metrics(ts_path):
    try:
        df = pd.read_parquet(ts_path)
        d = {str(r["metric"]).lower(): r["value"] for _, r in df.iterrows()}
        return {
            "cagr": float(d.get("cagr")) if "cagr" in d else None,
            "sharpe": float(d.get("sharpe")) if "sharpe" in d else None,
            "maxdd": float(d.get("maxdd")) if "maxdd" in d else None,
            "calmar": float(d.get("calmar")) if "calmar" in d else None,
        }
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--outfile", default="reports/run_manifest.jsonl")
    args = ap.parse_args()

    picks = pick_payloads(args.reports)
    if not picks:
        raise SystemExit("No portfolioV2* parquet payloads found in reports/")

    entry = {
        "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "host": platform.node(),
        "python": platform.python_version(),
        "cwd": os.getcwd(),
        "reports": os.path.abspath(args.reports),
        "artifacts": {},
    }

    for k, path in picks.items():
        st = os.stat(path)
        entry["artifacts"][k] = {
            "path": path,
            "size": st.st_size,
            "mtime": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
            "sha256": sha256_file(path),
        }

    if "tearsheet" in picks:
        m = try_tearsheet_metrics(picks["tearsheet"])
        if m:
            entry["tearsheet_metrics"] = m

    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    with open(args.outfile, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"✓ Appended manifest → {args.outfile}")


if __name__ == "__main__":
    main()

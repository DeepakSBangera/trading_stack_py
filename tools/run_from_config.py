import json
import os
import pathlib
import subprocess
import sys


def run(cmd):
    print("→", " ".join(cmd) if isinstance(cmd, (list, tuple)) else cmd)
    proc = subprocess.run(cmd, shell=isinstance(cmd, str))
    if proc.returncode != 0:
        sys.exit(proc.returncode)


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else r"config\baseline.json"
    if not os.path.exists(cfg_path):
        print(f"config not found: {cfg_path}")
        sys.exit(1)

    # Handle BOM (utf-8-sig)
    with open(cfg_path, encoding="utf-8-sig") as f:
        cfg = json.load(f)

    py = r".\.venv\Scripts\python.exe"
    if not os.path.exists(py):
        print(f"venv python not found at {py}")
        sys.exit(1)

    prices_root = cfg.get("prices_root", r"data_synth\prices")
    start = cfg.get("start", "2015-01-01")
    lookback = int(cfg.get("lookback", 126))
    top_n = int(cfg.get("top_n", 4))
    rebalance = cfg.get("rebalance", "ME")
    cost_bps = float(cfg.get("cost_bps", 10))
    outdir = cfg.get("outdir", "reports")

    # 1) Backtest (via wrapper → synthetic runner)
    wrap = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        r".\tools\Report-PortfolioV2.ps1",
        "-UniverseDir",
        prices_root,
        "-Start",
        start,
        "-Lookback",
        str(lookback),
        "-TopN",
        str(top_n),
        "-Rebalance",
        rebalance,
        "-CostBps",
        str(int(cost_bps)),
        "-OutDir",
        outdir,
    ]
    run(wrap)

    # 2) Find newest equity parquet (exclude weights/trades/tearsheet)
    outdir_p = pathlib.Path(outdir)
    cands = [
        p
        for p in outdir_p.glob("portfolioV2_*.parquet")
        if not (
            str(p.name).endswith("_weights.parquet")
            or str(p.name).endswith("_trades.parquet")
            or str(p.name).endswith("_tearsheet.parquet")
        )
    ]
    if not cands:
        print("No equity parquet found in reports/")
        sys.exit(1)
    latest_eq = max(cands, key=lambda p: p.stat().st_mtime)
    base = str(latest_eq).rsplit(".", 1)[0]

    # 3) Tearsheet (parquet + png)
    run([py, r".\tools\make_tearsheet.py", "--equity-csv", str(latest_eq)])

    # 4) W3/W4 (Parquet)
    run([py, r".\tools\compute_turnover.py", "--reports", outdir])
    run([py, r".\tools\voltarget_stops.py", "--reports", outdir])

    # 5) W5 (walk-forward PSR)
    run([py, r".\tools\walkforward_psr.py", "--reports", outdir, "--folds", "5"])

    # 6) Run manifest
    run([py, r".\tools\emit_run_manifest.py", "--reports", outdir])

    print("\n✓ Pipeline complete.")
    print("Artifacts (latest):")
    print(f"  equity:     {latest_eq}")
    print(f"  weights:    {base}_weights.parquet")
    print(f"  trades:     {base}_trades.parquet")
    print(f"  tearsheet:  {base}_tearsheet.parquet / .png")
    print(f"  W3:         {outdir}\\wk3_turnover_profile.parquet")
    print(f"  W4:         {outdir}\\wk4_voltarget_stops.parquet")
    print(f"  W5:         {outdir}\\wk5_walkforward.parquet")
    print(f"  manifest:   {outdir}\\run_manifest.jsonl")


if __name__ == "__main__":
    main()

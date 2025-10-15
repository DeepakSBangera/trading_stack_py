py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install git+https://github.com/DeepakSBangera/trading_stack_py.git@v1.1.5
trading-stack --ticker RELIANCE.NS --start 2015-01-01 --source synthetic

Trading · Pricing · Econometrics — Python-First (Low-Cost)

A beginner-friendly repo that grows week-by-week.

Trading: load prices, compute momentum/ATR, build a BUY list, backtest a simple momentum strategy.

Pricing (later): estimate price elasticity; generate price recommendations.

Econometrics (later): ADF/KPSS diagnostics + ARIMA baseline.

Quickstart (Windows / PowerShell)
cd /d F:\Projects\trading_stack_py
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt


Common runs:

# Daily signals (build features + BUY list)
python scripts\w1_build_entry_exit.py

# Simple backtest (momentum top-N, monthly rebalance)
python -m scripts.w2_backtest

# Dashboard (shows latest BUY list and portfolio tiles)
streamlit run app\dashboard.py


Outputs go to reports\
(e.g., reports\buylist_YYYY-MM-DD.csv and reports\backtests\<run>\...)

If imports like from src... fail, run modules from the repo root:
python -m scripts.w2_backtest

Week 3 — Costs & Churn Control (NEW)

We added:

A tiny incumbent bonus during ranking (small tie-breaker toward current holdings) to cut churn.

A turnover profile derived from your latest backtest.

A turnover guard that checks your weekly rebalance against policy caps.

Run it
# 1) Ensure you have a recent backtest run
python -m scripts.w2_backtest

# 2) Build turnover profile from the latest run
python scripts\w3_turnover.py
# -> writes: reports\backtests\<run>\turnover_by_rebalance.csv
#            reports\wk3_turnover_profile.csv (summary)

# 3) Set your policy caps (edit this if needed)
notepad reports\wk3_turnover_policy.json

# 4) Check the guard (flags breaches)
python scripts\w3_turnover_guard.py
# -> writes: reports\wk3_turnover_violations.csv


Default policy file (reports/wk3_turnover_policy.json):

{
  "policy_caps": {
    "target_avg_turnover_per_rebalance": 0.25,
    "target_p95_turnover": 0.45,
    "advisory_turnover_bps": 3500,
    "max_turnover_bps": 6000,
    "max_weekly_adds": 5,
    "max_weekly_drops": 5
  }
}

What you’ll see

reports/backtests/<run>/turnover_by_rebalance.csv – per-rebalance turnover (bps), adds/drops

reports/wk3_turnover_profile.csv – overall averages/percentiles

reports/wk3_turnover_violations.csv – lines for any WARN/HARD breaches or adds/drops caps

Data Flow (Week 0 → Week 1)
flowchart TD
  A[Raw price CSVs<br/>data/csv/*.csv] --> B[Loader & Feature Builder<br/>scripts/w1_build_entry_exit.py]
  F[Watchlists<br/>data/universe/*.csv] --> B
  G[Policy & Lists<br/>config/policy_w1.yaml<br/>config/lists/*] --> B
  B --> C[Per-ticker features<br/>data/features/*.parquet<br/>returns, ATR]
  C --> D[Rank & pick<br/>r12-1 momentum + list priority]
  D --> E[Reports<br/>reports/wk1_entry_exit_baseline.csv<br/>reports/buylist_YYYY-MM-DD.csv]

Turnover Pipeline (Week 3)
flowchart TD
  X[Backtest run<br/>python -m scripts.w2_backtest] --> Y[turnover_by_rebalance.csv]
  Y --> Z[wk3_turnover_profile.csv]
  P[Policy<br/>wk3_turnover_policy.json] --> G[Guard<br/>python scripts/w3_turnover_guard.py]
  Z --> G
  G --> V[wk3_turnover_violations.csv]


Notes:

The incumbent bonus is a small boost to current holdings (scale-aware, based on score std) to reduce unnecessary flips.

You can tune this in code (see INCUMBENT_BONUS_STD in scripts/w2_backtest.py).

Project Layout (high-level)
flowchart TB
  root([trading_stack_py])
  root --> gh[".github/"]
  root --> cfg["config/"]
  root --> data["data/"]
  root --> scripts["scripts/"]
  root --> reports["reports/"]
  root --> app["app/"]
  root --> readme[README.md]
  root --> pyproj[pyproject.toml]
  root --> reqs[requirements.txt]
  root --> env[sample.env]

  subgraph G[".github/"]
    ci["workflows/ci.yml"]
  end

  subgraph C["config/"]
    policy["policy_w1.yaml"]
    lists["lists/ (L2/L3)"]
  end

  subgraph D["data/"]
    csv["csv/*.csv (raw)"]
    prices["prices/*.parquet (clean)"]
    feats["features/*.parquet (features)"]
    uni["universe/*.csv (watchlists)"]
  end

  subgraph S["scripts/"]
    s1["w1_build_entry_exit.py"]
    s2["w1_signals_snapshot.py"]
    s3["w1_update_cfg.py"]
    s4["w2_backtest.py  (top-N momentum)"]
    s5["w3_turnover.py  (profile)"]
    s6["w3_turnover_guard.py (policy)"]
  end

  subgraph R["reports/"]
    r1["buylist_YYYY-MM-DD.csv"]
    r2["wk1_entry_exit_baseline.csv"]
    r3["backtests/<run> (csv/png)"]
    r4["wk3_turnover_profile.csv"]
    r5["wk3_turnover_violations.csv"]
  end

CI & Quality

Pre-commit (local): ruff, ruff-format, black, detect-secrets.

GitHub Actions: runs lint + format + tests on every push (.github/workflows/ci.yml).

Protect main by requiring PR + passing checks.

Troubleshooting

Activate venv (PowerShell): .\.venv\Scripts\Activate.ps1

Module imports fail? Run scripts via module from repo root: python -m scripts.w2_backtest

No plots? That’s fine—plots are optional unless matplotlib is installed.

Turnover guard says “file not found”? Run the backtest and w3_turnover.py first.


**W4 (vol-target + drawdown throttle):**
```bat
python -m scripts.w2_backtest
python scripts\w4_voltarget_stops.py

### W9 — Join W6 splits with W7 metrics

After producing a W6 dataset and a W7 modeling report, run:

```bash
python -m trading_stack_py.pipelines.evaluate_models \
  --w6-dir reports/W6/<your W6 folder> \
  --w7-dir reports/W7/<your W7 folder> \
  --tag REL_W9 \
  --outdir reports/W9

### Evaluate (W9)
```powershell
$w6 = (Get-ChildItem reports\W6 -Directory | Sort-Object LastWriteTime -Desc | Select -First 1).FullName
$w7 = (Get-ChildItem reports\W7 -Directory | Sort-Object LastWriteTime -Desc | Select -First 1).FullName
python -m trading_stack_py.pipelines.evaluate_models --w6-dir $w6 --w7-dir $w7 --tag REL_W9 --outdir reports/W9

# trading-stack-py

Minimal, reproducible trading research stack:
**data → signals → backtest → metrics**, plus CLIs for single-symbol
and top-N monthly rotation (momentum) portfolios.

- Python 3.11+
- Pre-commit (black, ruff) clean
- Works with synthetic or Yahoo data sources

## Install (editable)

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -e .


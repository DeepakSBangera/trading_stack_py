<!-- START OF README TO COPY -->

# Trading · Pricing · Econometrics — Python-First (Low-Cost)

A beginner-friendly repo that grows week-by-week.

- **Trading**: load prices, compute momentum/ATR, build a BUY list, backtest a simple momentum strategy.
- **Pricing (later)**: estimate price elasticity; generate price recs.
- **Econometrics (later)**: ADF/KPSS diagnostics + ARIMA baseline.

[![CI](https://github.com/DeepakSBangera/trading_stack_py/actions/workflows/ci.yml/badge.svg)](https://github.com/DeepakSBangera/trading_stack_py/actions/workflows/ci.yml)

---

## Quickstart (Windows)

```bat
cd /d F:\Projects\trading_stack_py
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
Daily signals (build features + BUY list):

bat
Copy code
python scripts\w1_build_entry_exit.py
Simple backtest (momentum top-N, monthly rebalance):

bat
Copy code
python -m scripts.w2_backtest
Dashboard (shows latest BUY list and basic portfolio view):

bat
Copy code
streamlit run app\dashboard.py
Outputs go to reports\ (e.g., reports\buylist_YYYY-MM-DD.csv and reports\backtests\...).

What’s inside (Week 0 → Week 1)
mermaid
Copy code
flowchart TD
  A[Raw price CSVs<br/>data/csv/*.csv] --> B[Loader & Feature Builder<br/>scripts/w1_build_entry_exit.py]
  F[Watchlists<br/>data/universe/*.csv] --> B
  G[Policy & Lists<br/>config/policy_w1.yaml<br/>config/lists/*] --> B
  B --> C[Per-ticker features<br/>data/features/*.parquet<br/>returns, ATR]
  C --> D[Rank & pick<br/>momentum score + list priority]
  D --> E[Reports<br/>reports/wk1_entry_exit_baseline.csv<br/>reports/buylist_YYYY-MM-DD.csv]

  subgraph Tooling
    H[Pre-commit (local)<br/>ruff, black, secrets]
    I[CI on GitHub<br/>.github/workflows/ci.yml]
  end
  H -.checks before commit.-> B
  I -.checks on push.-> E
Project layout (high-level)
mermaid
Copy code
flowchart TB
  root([trading_stack_py])

  %% top-level
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

  %% .github
  subgraph G[".github/"]
    direction TB
    ci["workflows/ci.yml"]
  end
  gh --> ci

  %% config
  subgraph C["config/"]
    direction TB
    policy["policy_w1.yaml"]
    lists["lists/"]
  end
  cfg --> policy
  cfg --> lists
  subgraph L["config/lists/"]
    direction TB
    l2["list2_conviction.csv"]
    l3["list3_quality.csv"]
  end
  lists --> l2
  lists --> l3

  %% data
  subgraph D["data/"]
    direction TB
    csv["csv/*.csv  (raw price files)"]
    prices["prices/*.parquet  (clean prices)"]
    feats["features/*.parquet  (computed features)"]
    uni["universe/*.csv  (watchlists)"]
  end
  data --> csv
  data --> prices
  data --> feats
  data --> uni

  %% scripts
  subgraph S["scripts/"]
    direction TB
    s1["w1_build_entry_exit.py  (compute momentum & ATR, make buylist)"]
    s2["w1_signals_snapshot.py  (summarize signals)"]
    s3["w1_update_cfg.py  (update config)"]
    s4["w2_backtest.py  (simple top-N momentum backtest)"]
  end
  scripts --> s1
  scripts --> s2
  scripts --> s3
  scripts --> s4

  %% reports
  subgraph R["reports/"]
    direction TB
    r1["wk1_entry_exit_baseline.csv"]
    r2["buylist_YYYY-MM-DD.csv"]
    r3["backtests/<run>/ (csv, png)"]
  end
  reports --> r1
  reports --> r2
  reports --> r3
CI & quality checks
Pre-commit runs locally: ruff (lint), ruff-format, black, detect-secrets.

GitHub Actions runs the same checks + tests on every push: .github/workflows/ci.yml.

If hooks block your commit, read the message and apply the suggested fixes (or run pre-commit run -a).

Notes
On Windows PowerShell, activate the venv with:

powershell
Copy code
.\.venv\Scripts\Activate.ps1
If imports like from src... fail, run scripts via module form from repo root:

powershell
Copy code
python -m scripts.w2_backtest
<!-- END OF README TO COPY -->
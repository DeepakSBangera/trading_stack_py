A beginner-friendly repo that grows week-by-week:

Trading: load prices, compute momentum/ATR, build a BUY list, backtest a simple momentum strategy.

Pricing (later): estimate price elasticity; generate price recommendations.

Econometrics (later): ADF/KPSS diagnostics + ARIMA baseline.

Quickstart (Windows / PowerShell)
cd /d F:\Projects\trading_stack_py
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt


Daily signals (build features + BUY list):

python scripts\w1_build_entry_exit.py


Simple backtest (momentum top-N, monthly rebalance):

python -m scripts.w2_backtest


Dashboard (shows latest BUY list and basic portfolio view):

streamlit run app\dashboard.py


Outputs land in reports\ (e.g., reports\buylist_YYYY-MM-DD.csv and reports\backtests\...).

What’s inside (Week 0 → Week 1)
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
  H -. checks before commit .-> B
  I -. checks on push .-> E

Project layout (high-level)
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
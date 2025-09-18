# Trading · Pricing · Econometrics — Python-First (Low-Cost)

## What this repo gives you
- Trading: fetch data, compute indicators/signals, write a buy list, view in Streamlit.
- Pricing: estimate log–log price elasticity; write price recommendations.
- Econometrics: run ADF/KPSS diagnostics + ARIMA baseline forecasts.

## Quickstart (Windows, F:)
```bat
cd /d F:\Projects\trading_stack_py
python -m venv .venv
.\.venv\Scripts ctivate
pip install -r requirements.txt

python run_daily.py
streamlit run app\dashboard.py
```

### Pricing (later)
Place `pricing\data\transactions.csv` with columns:
`date,product_id,price,qty,promo_flag,cost`
Then run:
```bat
python pricing\run_pricing.py
```

### Econometrics (later)
Put `econo\timeseries\*.csv` with columns: `date,value`
Then run:
```bat
python run_econo.py
```

[![CI](https://github.com/DeepakSBangera/trading_stack_py/actions/workflows/ci.yml/badge.svg)](https://github.com/DeepakSBangera/trading_stack_py/actions/workflows/ci.yml)

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

flowchart TB
  root([trading_stack_py])

  %% top-level
  root --> gh(".github/")
  root --> cfg("config/")
  root --> data("data/")
  root --> scripts("scripts/")
  root --> reports("reports/")
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
    uni["universe/  (watchlists)"]
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
  end
  scripts --> s1
  scripts --> s2
  scripts --> s3

  %% reports
  subgraph R["reports/"]
    direction TB
    r1["wk1_entry_exit_baseline.csv"]
    r2["buylist_YYYY-MM-DD.csv"]
  end
  reports --> r1
  reports --> r2

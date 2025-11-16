# Trading Stack — Living Blueprint v1 (Operational Plan & Tracker)

> **Purpose (living doc):** One place to see vision, scope, architecture, data schemas, operational knobs, risks, backlog, and progress. Updated every session. The goal is a robust app that compounds wealth at high risk-adjusted rates using a disciplined quant + fundamentals approach, with production-grade hygiene.

---

## 1) Vision & Success Criteria
**North Star:** Durable, tax-aware, capacity-aware portfolio engine for Indian equities (expandable to multi-asset) that targets **net CAGR 18–25%** with **Sharpe ≥ 1.0**, **Calmar ≥ 0.8**, controlled drawdowns (≤30% target), and codified governance.
**Philosophy:** Blend cross-sectional momentum + quality & pricing-power filters (quant core) with a small discretionary sleeve (fundamental conviction), under strict risk, capacity, and change-control.
**Operational Tenets:**
- Point-in-time (PIT) data integrity; no look-ahead.
- Canary → Promote → Freeze lifecycle for changes.
- Exposure caps (sector/style/factor) + risk budgeting (vol targeting, Kelly-fraction limits).
- After-tax, after-cost evaluation.

---

## 2) Current Status (snapshot)
**Repo root:** F:\Projects\trading_stack_py  
**Environment:** Windows, PowerShell, Python venv .venv  
**Last session outcome:**
- Factors pipeline online: sector roll, momentum (12-1 proxy), quality (inverse downside vol). Artifacts in 
eports/.
- Modularized modules/* structure, hygiene scripts, Living Tracker in docs/.
- Kit self-test scripts pass (imports + sanity numbers).
**Near-term gaps:**
- Canonical price panel & universe IO (schema + loaders) to complete portfolio W6.
- Dashboards beyond index; attribution tiles; IC half-life tracking.

---

## 3) Architecture Map (logical)
Layers
1. DataPipe — Sources, schemas, loaders, lineage, PIT checks, caching (parquet).
2. AlphaFactors — Signals & style proxies (momentum, quality, sector exposures), signal QC.
3. Portfolio — Optimizers (EW vs LW-GMV), constraints (ADV%, sector caps), exposure penalties.
4. RiskSizing — Vol targeting, ATR stops, drawdown throttle, Kelly fraction.
5. Monitoring — KPIs (CAGR/Sharpe/Calmar/PF), attribution, IC half-life, alerts.
6. Governance — Change control, kill-switches, canary, capstone freeze.
7. Dashboards — HTML/PNG reports; index page; drill-downs.

Execution surfaces
- CLI Python entrypoints in modules/*/*.py
- PowerShell wrappers in scripts/*.ps1

---

## 4) Directory & Key Files (where knobs live)
modules/
  DataPipe/      data_io.py, w3_turnover_guard.py, data_lineage.md
  AlphaSignals/  signals.py, w1_build_entry_exit.py, w1_signals_snapshot.py
  AlphaFactors/  build_factors.py, make_factor_exposures.py, exposures.py
  Portfolio/     w6_portfolio_compare.py, build_portfolio.py, capacity_policy.md
  RiskSizing/    w4_voltarget_stops.py, w4_atr_stops.py, w4_risk.yaml
  Monitoring/    attribution.py, dashboard.py, ic_promotion_rules.md
  Dashboards/    make_report_index.py, plot_tearsheet_v2.py
tradingstack/
  metrics/       sharpe.py, sortino.py, calmar.py, drawdown.py, omega.py
  factors/       __init__.py (proxies + loaders)
config/          run.yaml, sector_mapping.csv
reports/         *.csv, *.parquet, index.html
scripts/         *.ps1 wrappers (build, view, test, clean)
docs/            living_tracker.md/csv, ARCHITECTURE.md

Config knobs:
- config/run.yaml — run windows, rebalance, risk targets, IO roots
- config/sector_mapping.csv — ticker→sector map
- modules/RiskSizing/w4_risk.yaml — vol target, ATR, DD throttle, Kelly
- modules/Portfolio/capacity_policy.md — ADV%, min ₹ADV, sector caps

---

## 5) Data Schemas (canonical)
Prices panel (EOD): date, open, high, low, close, volume; index UTC, increasing, unique.
Universe: ticker, active, sector, adv_inr, freefloat, ...
Portfolio: reports/portfolioV2_*.csv → date, nav, equity, cash, ...
Weights: reports/portfolioV2_*_weights.csv → date, TICKER...

Contracts: all UTC; no tz-naive joins; dedupe by keep-last then sort.

---

## 6) Pipelines (inputs→outputs)
A) Factor Exposures
- Entry: tools/make_factor_exposures.py
- Inputs: portfolioV2_*.csv, *_weights.csv, sector_mapping.csv
- Outputs: reports/factor_exposures.parquet/csv/txt
- Knobs: --window

B) Portfolio Compare (W6)
- Entry: modules/Portfolio/w6_portfolio_compare.py
- Inputs: universe.csv, prices root, start date
- Output: reports/wk6_portfolio_compare.csv
- Knobs: --lookback --rebalance --cost-bps + capacity caps

C) Reports Index
- Entry: modules/Dashboards/make_report_index.py
- Output: reports/index.html

---

## 7) KPIs & Observability
Compounding: CAGR, Sharpe, Calmar, PF
Risk: Max DD path, realized vol vs target, exposure heatmap
Alpha quality: IC by sleeve, IC half-life, breadth
Drag: Turnover, slippage p50/p95, tax drag, capacity utilization
Governance: canary days, kill-switch tests, lineage completeness

Artifacts (TBD): reports/ic_timeseries.csv, signal_half_life.csv, wk17_ops_attrib.csv

---

## 8) Governance & Change Control
Lifecycle: Sandbox → Canary (1–5%) → Promotion (W11/W37) → Capstone Freeze (W20)
Kill-Switches: conditions (VaR breach, DD>8%, data outage, slippage>budget) → actions (halt adds, de-risk 25–50%, switch Aggressive→Base, restrict to ETFs, notify & record)
Files: docs/kill_switch_matrix.md, config/kill_switch.yaml, reports/kill_switch_tests.csv
Data Lineage: run manifest, artifact hashes, retention policy

---

## 9) Backlog & Hours
Completed (~14h): modularization, hygiene, factor exposures, metrics, kit tests, review zips, report index.

Core To-Go (~25–41h):
1) Data IO Canonicalization (4–6h)
2) Signals v1 (4–6h)
3) Portfolio W6 (6–8h)
4) Risk & Stops (4–6h)
5) Monitoring v1 (4–6h)

Scale Upgrades (~12–17h): partitioning, batch runners, caching, manifesting, report sweeps

---

## 10) Risks & Weaknesses
Data drift/schema breaks → schema guards, PIT tests, lineage hashes
Capacity & slippage → ADV caps, TCA tiles, scheduled execution
Overfitting → Walk-forward + DSR, PBO checks, canary discipline
Tax drag → tax-aware lot scheduling (W42), hysteresis

---

## 11) Operating Rhythm
1) Branch session/*
2) Run module chain
3) Verify KPIs/gates; update this doc
4) Commit + tag + backup

---

## 12) Changelog
- 2025-10-31 — v1 created; factors pipeline stable; kit test OK; next: Data IO + Signals v1

---

## 13) Update Rules
Edit docs/LIVING_BLUEPRINT.md every session; update Status/Backlog/Hours + Changelog.

---

## 14) Glossary
PIT, DSR, LW-GMV, ADV%

Appendix A — Locators:
- Factor exposures: tools/make_factor_exposures.py
- Signals baseline: modules/AlphaSignals/w1_build_entry_exit.py
- Portfolio compare: modules/Portfolio/w6_portfolio_compare.py
- Risk knobs: modules/RiskSizing/w4_risk.yaml
- Dash index: modules/Dashboards/make_report_index.py
- Kit tests: scripts/Test-Kit.ps1

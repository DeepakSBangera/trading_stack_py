# BUILD TRACKER — trading_stack_py (Living Document)
**Canonical, cumulative record.** If it’s not here, it doesn’t exist.  
Owner: you + ChatGPT (pair) • Repo: `F:\Projects\trading_stack_py` • Venv: `.\.venv`

---

## A) Strategy & Rules (always follow)
1) **Tracker-first:** Add/update purpose, API, schema, runbook, and changelog **here** before coding.  
2) **Separation of concerns:**  
   - `tradingstack/` → reusable library logic only.  
   - `tools/` → orchestrators/CLI; no metric math here.  
   - `scripts/*.ps1` → convenience (create/open files, run pipelines).  
3) **Date discipline:** normalize to tz-naive `DatetimeIndex` named `date` before rolling/joins.  
4) **Contracts:** every new parquet’s schema is recorded here. Changes require a version note in **M) Changelog**.  
5) **Compat over breakage:** adapt names via a compatibility layer; don’t silently delete/rename public APIs.  
6) **Repro:** every report has a runbook step here.

---

## B) Project Context
- **Repo root:** `F:\Projects\trading_stack_py`  
- **Venv:** `.\.venv`  
- **Primary data:** `.\data_synth\prices`  
- **Reports (live):** `.\reports`  
- **Reports (archive):** `.\reports\archive`  
- **Config:** `.\config`  
- **Source packages:** `.\tradingstack\...`  
- **Orchestrators:** `.\tools`  
- **PowerShell helpers:** `.\scripts`  
- **Docs (this file):** `.\docs\BUILD_TRACKER.md`

---

## C) What’s done (✅ Verified so far)
- Parquet tear-sheet comparator  
- Automated report archival  
- `add_date_if_missing` repair util  
- Date sanity checks via `tools\parquet_has_date.py`  
- Modular pandas loader: `tradingstack.io.equity`  
- Risk metrics modules: `sharpe`, `drawdown`, `sortino`, `calmar`, `omega`  
- DSR + PBO evaluator (wk5)  
- Promotion decision gate  
- Normalized weights transformation  
- Sector attribution engine (ticker & portfolio returns)  
- Attribution parity verification (abs_diff ~ 0)  
- Omega integrated as official metric  
- Attribution output now Parquet + txt note  
- Backtesting intermediate reports consistent

**Verified outputs**
- `reports/portfolio_v2.parquet` (has `date`, `_nav`)  
- `reports/attribution_ticker.parquet`  
- `reports/attribution_portfolio_returns.parquet`  
- `reports/attribution_parity.txt`  
- `reports/weights_v2_norm.parquet`

**Known tech details**
- `fillna(method="ffill")` deprecated → use `.ffill()`  
- TZ conversions fragile if index not `DatetimeIndex` → normalize early  
- Attribution parity robust (abs diff ≈ 0)  
- `_nav` exists

---

## D) Architecture Map
- `tradingstack/io/equity.py` → IO loading & tidy DataFrame creation  
- `tradingstack/metrics/*.py` → **pure** metric math (point & rolling)  
- `tradingstack/utils/*.py` → cross-cutting helpers (e.g., date normalization)  
- `tools/*.py` → read config → call library → write outputs → print summaries  
- `scripts/*.ps1` → create/open files, run pipelines, print summaries  
- `reports/` → generated artifacts only; archive under `reports/archive/`

---

## E) Public API (stable import surface)
Use these names going forward (internals may vary; compat adapts):
- Point metrics: `sharpe_ratio`, `sortino_ratio`, `max_drawdown`, `calmar_ratio`, `omega_ratio`
- Rolling: `rolling_volatility`, `rolling_sharpe`, `rolling_sortino`, `rolling_drawdown`, `trend_regime`, `compute_rolling_metrics_from_nav`
- IO: `tradingstack.io.equity` loader functions
- Attribution: exports in `tradingstack.metrics.attribution` (lock here once final)

> If internal function names differ, map them in **I) API Compatibility Mapping**.

---

## F) Data Contracts (schemas)
**F.1 `reports/portfolio_v2.parquet`**  
- Required: `date: datetime64[ns]` (tz-naive; index or column), `_nav: float`  
- Constraints: `_nav` > 0; unique, ascending `date`

**F.2 `reports/weights_v2_norm.parquet`**  
- Required: `date`, `ticker`, `weight_norm`  
- Constraint: by `date`, sum(`weight_norm`) ≈ 1.0 (±1e-6)

**F.3 `reports/attribution_ticker.parquet`**  
- Required: `date`, `ticker`, `ret_contrib` (or agreed stable name)

**F.4 `reports/attribution_portfolio_returns.parquet`**  
- Required: `date`, `portfolio_ret`

**F.5 (S2) `reports/rolling_metrics.parquet`**  
- Index: `date`  
- Columns: `rolling_vol`, `rolling_sharpe`, `rolling_sortino`, `rolling_mdd` (≤ 0), `regime` (int8 in {-1, 0, 1})

**F.6 `reports/rolling_metrics_summary.txt`**  
- Snapshot: rows, start/end, latest metrics, NaN counts

---

## G) Runbooks
**RB-S2-002: Plot Rolling Metrics**
1) Pre-req: eports/rolling_metrics.parquet present (run RB-S2-001).
2) Run: pwsh .\scripts\Plot-Rolling.ps1
3) Output: eports/rolling_metrics.png (5 panels: vol, sharpe, sortino, maxDD, regime)

## H) Roadmap (living)
- **Session 2: Rolling Metrics Engine (NOW)**  
  - [x] Date normalization utility (`utils/dates.py`)  
  - [x] Rolling vol/sharpe/sortino/drawdown + regimes (`metrics/rolling.py`)  
  - [x] Orchestrator + config + PS helper  
  - [ ] Unit tests & smoke checks (flat NAV, short windows)  
- **Session 3: Factor Exposures** — sector betas; momentum proxy; quality proxy  
- **Session 4: Tearsheet v2** — link rolling + exposures + attribution; PNG/HTML  
- **Session 5: Reporting layer** — standardized summaries, paths, archive policy  
- **Session 6: Modularity & cleanup** — dead code scan, naming normalization, packaging polish

---

## I) API Compatibility Mapping (actual → canonical)
- 	radingstack.metrics.sharpe: (actual: TBD) → **sharpe_ratio**
- 	radingstack.metrics.sortino: (actual: TBD) → **sortino_ratio**
- 	radingstack.metrics.drawdown: (actual: TBD) → **max_drawdown**
- 	radingstack.metrics.calmar: (actual: TBD) → **calmar_ratio**
- 	radingstack.metrics.omega: (actual: TBD) → **omega_ratio**

## J) Quality Gates
- **Schema Gate:** parquets match Section F  
- **Index Gate:** `date` unique, sorted, tz-naive  
- **NaN Gate:** no NaNs in `_nav`; rolling NaNs only from warm-up windows  
- **Attribution Parity Gate:** abs diff ~ 0; variance logged  
- **Promotion Gate:** all above pass → archive snapshot

---

## K) Gaps / Next Actions (current)
1) Confirm/record **actual function names** → fill **Section I**.  
2) Add tests for `normalize_date_index()` & rolling edge cases.  
3) Document factor exposure schemas before coding (S3).  
4) Mark safe-to-prune legacy files after audit (list in **L**).

---

## L) Prune / Archive Candidates (to be filled post-audit)
- (path) — reason  
- (path) — reason

---

## M) Changelog
- **2025-10-26** — BUILD_TRACKER created inside repo; strategy & gates documented
- **2025-10-26** — Locked tracker strategy; added helpers




## I) API Compatibility Mapping (actual â†’ canonical)
- tradingstack.metrics.calmar: (actual: calmar_ratio) â†’ **calmar_ratio**
- tradingstack.metrics.drawdown: (actual: max_drawdown) â†’ **max_drawdown**
- tradingstack.metrics.omega: (actual: omega_ratio) â†’ **omega_ratio**
- tradingstack.metrics.sharpe: (actual: _to_series) â†’ **sharpe_ratio**
- tradingstack.metrics.sortino: (actual: _as_returns_series) â†’ **sortino_ratio**
- **2025-10-26** — Added RB-S2-002 and plot exporter for rolling metrics
- **2025-10-26** — Tagged s2-complete; (optional) committed rolling PNG



## G) Factor Exposures (rolling sector, momentum, quality)
**Latest snapshot**

```text
Rows: 261   Start: 2025-01-01   End: 2025-12-31

Last date: 2025-12-31
  Momentum (12-1) proxy: NaN
  Quality (inv downside vol): 16.1663

Top sector exposures (rolling avg weights):
  - sector_Information Technology: 0.5000
  - sector_Energy: 0.2500
  - sector_Financials: 0.2500
```

## H) Tearsheet v2 Snapshot
**Latest snapshot (Tearsheet v2)**

File: reports\tearsheet_v2.png
HTML: reports\tearsheet_v2.html

## I) Reporting Layer (Session 5)
Daily report index: reports\index_daily.html
RollingWindow: 84
Start: 2025-01-01
UniverseCsv: config/universe.csv
- **2025-10-27** — Session 4: Tearsheet v2 generated


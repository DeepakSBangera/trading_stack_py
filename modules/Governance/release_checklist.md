\# Release Checklist (MVP v1.0)



\## 0) Meta

\- \[ ] Branch merged to main build branch (`chore/canonize-v1-31` or `main`)

\- \[ ] Working tree clean: `git status` shows no changes

\- \[ ] Pre-commit all green: `pre-commit run --all-files`



\## 1) Functionality gates (MVP)

\- \[ ] Data sources: synthetic + local CSV + Yahoo (with retry/backoff)

\- \[ ] Indicators: SMA, OBV, Supertrend

\- \[ ] Signals: config-driven MA (Close>SMA or fast>slow), CLI `--use\_crossover`

\- \[ ] Backtester: ENTRY/EXIT → Equity/Return; cost\_bps applied

\- \[ ] Metrics: CAGR, Sharpe, MaxDD, Calmar, PSR util

\- \[ ] Scripts: `runner\_cli.py`, `grid\_search.py`, `walkforward\_psr.py`, `report\_polish.py`



\## 2) Repro artifacts

\- \[ ] Single run:

&nbsp; - Command used:

&nbsp;   ```bash

&nbsp;   python scripts/runner\_cli.py --ticker RELIANCE.NS --start 2015-01-01 --source synthetic

&nbsp;   ```

&nbsp; - Output exists: `reports/run\_RELIANCE\_NS.csv`

\- \[ ] Grid search:

&nbsp; - Command used:

&nbsp;   ```bash

&nbsp;   python scripts/grid\_search.py --ticker RELIANCE.NS --start 2015-01-01 --source synthetic

&nbsp;   ```

&nbsp; - Output exists: `reports/grid\_RELIANCE\_NS.csv`

\- \[ ] Walk-forward + PSR:

&nbsp;   ```bash

&nbsp;   python scripts/walkforward\_psr.py --ticker RELIANCE.NS --start 2015-01-01 --source synthetic --train\_years 3 --test\_years 1

&nbsp;   ```

&nbsp; - Output exists: `reports/walkforward\_RELIANCE\_NS.csv`

\- \[ ] Summary roll-up:

&nbsp;   ```bash

&nbsp;   python scripts/report\_polish.py

&nbsp;   ```

&nbsp; - Output exists: `reports/build\_summary.csv`



\## 3) Docs

\- \[ ] `docs/CHANGELOG.md` updated for v1.0

\- \[ ] (Optional) `docs/process\_notes.md` add a short “MVP cut” line



\## 4) Tag \& Release

\- \[ ] Create annotated tag `v1.0`

\- \[ ] Push tag to origin

\- \[ ] (Optional) GitHub release created, attach CSVs from `reports/` and paste changelog notes



\## 5) Post-release hygiene

\- \[ ] Open follow-up issue(s): Yahoo caching/cool-down; add `docs/release\_checklist.md` to pipeline

\- \[ ] Protect branch / set CI status checks (future)




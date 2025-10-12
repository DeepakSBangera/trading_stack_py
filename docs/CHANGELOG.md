\# Changelog



All notable changes to this project will be documented in this file.



\## \[v1.0] - 2025-10-12

\### Highlights (MVP)

\- \*\*Data Loader\*\*: `auto|local|yahoo|synthetic` with backoff and CSV caching.

\- \*\*Indicators\*\*: SMA, OBV, Supertrend (configurable).

\- \*\*Signals\*\*: Config-driven MA logic; runtime `--use\_crossover` override.

\- \*\*Backtester\*\*: ENTRY/EXIT → Equity \& Return series; trading costs via `cost\_bps`.

\- \*\*Metrics\*\*: CAGR, Sharpe, MaxDD, Calmar; PSR utility aligned with caller.

\- \*\*CLI \& Scripts\*\*:

&nbsp; - `scripts/runner\_cli.py` single-run driver (saves run CSV + prints stats)

&nbsp; - `scripts/grid\_search.py` parameter sweep (leaderboard CSV)

&nbsp; - `scripts/walkforward\_psr.py` 3y/1y example with PSR column

&nbsp; - `scripts/report\_polish.py` roll-up `build\_summary.csv`

\- \*\*Tag anchor\*\*: prior milestone `v1.31-build-milestone` retained.



\### Notes

\- Yahoo path is subject to rate limiting; synthetic \& local paths are stable.

\- See `config/strategy.yml` for MA defaults and portfolio params.



\## \[Unreleased]

\- Caching for Yahoo downloads

\- Portfolio-level compare expansion




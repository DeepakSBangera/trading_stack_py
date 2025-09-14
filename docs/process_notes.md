# Process & Discipline — Notes (W0)

**Key ideas**
- Clear hypothesis → Signals → Portfolio → Review loop.
- Pre-defined **policy gates** protect deployments (CAGR / MDD / Sharpe / PF).
- Separation of research vs. evaluation vs. production; use checklists.
- Benchmark against market; measure *edge* and *process consistency*.

**Actionables for this repo**
- Keep week-level gates in `reports/wk0_gates.csv` (then wk1_gates.csv, …).
- Maintain market benchmarks in `reports/benchmarks.csv`.
- Treat `run_daily.py` outputs as production artifacts; review weekly.
- Use pre-commit (ruff/black/secrets) to enforce discipline.

**W0 status**
- Pipeline runs end-to-end; data via CSV; empty buylist is acceptable for W0.
- Next: wire portfolio/backtest metrics to populate gates automatically.

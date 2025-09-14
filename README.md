# Trading Plan — W0 Bootstrap

**Goal (W0):** Benchmarks & policy gates. Repo+env ready; paper-trade only.

## Quickstart
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Unix: source .venv/bin/activate
pip install -r requirements.txt
pre-commit install && pre-commit run --all-files
pytest -q
```

See `docs/benchmarks.md` and `docs/policy_gates.md`. Track gates in `reports/wk0_gates.csv`.

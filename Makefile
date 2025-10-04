.PHONY: init check-structure ci-smoke qc demo-logging

init:
	python -m venv .venv || true
	pip install -r requirements.txt
	pre-commit install

check-structure:
	python scripts/check_structure.py > reports/artifacts/tree_snapshot.txt
	@echo "OK"

ci-smoke:
	pytest -q

qc:
	python scripts/qc_example.py

demo-logging:
	python scripts/demo_logging.py
w10:
\tpython scripts/w10_arimax.py --data-glob "data/csv/*.csv" --out "reports/wk10_forecast_eval.csv" --order "1,1,1"

w11:
\tpython scripts/w11_alpha_blend.py --data-glob "data/csv/*.csv" --out "reports/wk11_alpha_blend.csv"

w11:
\tpython scripts/w11_alpha_blend.py --out "reports/wk11_alpha_blend.csv"





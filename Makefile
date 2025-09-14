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

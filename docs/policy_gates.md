# Policy Gates — W0 Sign-off

## Risk Limits
- Leverage: 1.0x (paper trade), 0.0x live until W16
- Sector cap: 30% of portfolio; single name: 10%
- Stop rules: hard stop -3% per position; portfolio DD alert -10%

## Execution & Governance
- Paper trade flags enforced; no broker keys in env
- Pre-commit: ruff, black, gitleaks
- CI: lint, unit tests, coverage>=60% baseline

## Evidence Artifacts
- `reports/artifacts/tree_snapshot.txt`
- `reports/artifacts/setup_log.txt`
- `reports/artifacts/ci_summary.json`
- `reports/artifacts/secret_scan.json`
- `reports/artifacts/data_qc_report.csv`
- `reports/artifacts/logging_demo.json`

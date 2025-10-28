[CmdletBinding()]
param(
  [switch]$Fast
)
$ErrorActionPreference = 'Stop'
Set-Location 'F:\Projects\trading_stack_py'

$venv = '.\.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $venv)) { throw 'venv python not found' }

# Make project importable
$repo = (Resolve-Path '.').Path
$env:PYTHONPATH = if ($env:PYTHONPATH) { "$repo;$($env:PYTHONPATH)" } else { $repo }

Write-Host "== Quick-CI =="
# 1) Schema dumper
& $venv .\tools\dump_parquet_schemas.py | Out-Null

# 2) Rolling metrics (fast if requested)
if ($Fast) {
  & $venv .\tools\make_rolling_metrics.py --window 63
} else {
  & $venv .\tools\make_rolling_metrics.py --window 126
}
if ($LASTEXITCODE -ne 0) { throw 'rolling metrics failed' }

# 3) Factor exposures
if (Test-Path -LiteralPath .\tools\make_factor_exposures.py) {
  & $venv .\tools\make_factor_exposures.py
}

# 4) Plot tearsheet
& $venv .\tools\plot_tearsheet_v2.py

Write-Host "[OK] Quick-CI completed."

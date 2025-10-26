# scripts/Run-DSR.ps1
# Runs DSR/PBO on the walk-forward equity file and writes two artifacts:
#  - reports\wk5_walkforward_dsr.parquet
#  - reports\wk5_walkforward_dsr.csv

param(
  [string]$PythonPath = ".\.venv\Scripts\python.exe",
  [string]$Equity = ".\reports\wk5_walkforward.parquet",
  [string]$OutParquet = ".\reports\wk5_walkforward_dsr.parquet",
  [string]$OutCsv = ".\reports\wk5_walkforward_dsr.csv"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $PythonPath)) { throw "Python venv not found at $PythonPath" }
if (-not (Test-Path $Equity))     { throw "Equity parquet not found: $Equity" }

& $PythonPath ".\tools\dsr_pbo.py" --equity $Equity --out $OutParquet --csv $OutCsv
if ($LASTEXITCODE -ne 0) { throw "dsr_pbo.py failed: $LASTEXITCODE" }

Write-Host "DSR/PBO artifacts written."
Write-Host " - $OutParquet"
Write-Host " - $OutCsv"

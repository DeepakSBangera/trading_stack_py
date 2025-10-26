param(
  [Parameter(Mandatory=$true)][string]$A,
  [Parameter(Mandatory=$true)][string]$B,
  [string]$OutDir = ".\reports",
  [string]$PythonPath = ".\.venv\Scripts\python.exe"
)
$ErrorActionPreference = "Stop"
if (-not (Test-Path $PythonPath)) { throw "Python not found: $PythonPath" }
$compare = ".\tools\compare_runs.py"
if (-not (Test-Path $compare)) { throw "Comparator missing: $compare" }
& $PythonPath -X utf8 $compare --a "$A" --b "$B" --out "$OutDir"
if ($LASTEXITCODE -ne 0) { throw "compare_runs.py failed ($LASTEXITCODE)" }
Write-Host "Compare written to $OutDir (compare_runs.*)" -ForegroundColor Green

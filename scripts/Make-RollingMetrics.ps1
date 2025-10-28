[CmdletBinding()]
param(
  [int]$Window = 63,
  [switch]$Open
)
$ErrorActionPreference = 'Stop'
Set-Location 'F:\Projects\trading_stack_py'

$venv = '.\.venv\Scripts\python.exe'
$tool = '.\tools\make_rolling_metrics.py'
$outp = '.\reports\rolling_metrics_summary.txt'
if (-not (Test-Path -LiteralPath $venv)) { throw 'venv python not found' }
if (-not (Test-Path -LiteralPath $tool)) { throw 'missing tools\make_rolling_metrics.py' }

# Make tradingstack importable
$repo = (Resolve-Path '.').Path
$env:PYTHONPATH = if ($env:PYTHONPATH) { "$repo;$($env:PYTHONPATH)" } else { $repo }

# Try with --window, then fallback to no-arg if not supported
& $venv $tool --window $Window
if ($LASTEXITCODE -ne 0) {
  & $venv $tool
  if ($LASTEXITCODE -ne 0) { throw 'make_rolling_metrics failed' }
}

if ($Open -and (Test-Path -LiteralPath $outp)) { Start-Process -FilePath $outp }
Write-Host 'Rolling metrics rebuilt.'

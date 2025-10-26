# Plot-Rolling.ps1 — export rolling metrics PNG and open it
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Ensure we're in the correct repo
Set-Location F:\Projects\trading_stack_py

$in  = ".\reports\rolling_metrics.parquet"
$out = ".\reports\rolling_metrics.png"

if (-not (Test-Path $in)) {
  throw "Input not found: $in. Run: .\.venv\Scripts\python.exe .\tools\make_rolling_metrics.py"
}

.\.venv\Scripts\python.exe .\tools\plot_rolling_metrics.py

if (Test-Path $out) {
  Write-Host "[OK] Opening $out"
  Start-Process $out
} else {
  Write-Host "[WARN] PNG not found at $out"
}

[CmdletBinding()]
param(
  [string]$Reports = "reports"
)
$py = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "Python venv not found at $py" }

Write-Host "Running W3 (turnover → Parquet)..." -ForegroundColor Cyan
try {
  & $py .\tools\compute_turnover.py --reports $Reports
} catch {
  Write-Warning "W3 turnover script failed: $($_.Exception.Message)"
}

Write-Host "Running W4 (vol & stops → Parquet)..." -ForegroundColor Cyan
try {
  & $py .\tools\voltarget_stops.py --reports $Reports
} catch {
  Write-Warning "W4 voltarget script failed: $($_.Exception.Message)"
}

Write-Host "`nDone. Parquet outputs (if present):" -ForegroundColor Green
$paths = @(
  (Join-Path $Reports "wk3_turnover_profile.parquet"),
  (Join-Path $Reports "wk4_voltarget_stops.parquet")
)
$present = @()
foreach ($p in $paths) { if (Test-Path $p) { $present += Get-Item $p } }
if ($present.Count -gt 0) {
  $present | Select-Object Name, Length, LastWriteTime | Format-Table -AutoSize
} else {
  Write-Host "No outputs found." -ForegroundColor Yellow
}

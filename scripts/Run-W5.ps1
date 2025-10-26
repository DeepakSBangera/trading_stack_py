[CmdletBinding()]
param(
  [string]$Reports = "reports",
  [int]$Folds = 5
)
$py = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "Python venv not found at $py" }

Write-Host "W5: Walk-forward (folds=$Folds)..." -ForegroundColor Cyan
& $py .\tools\walkforward_psr.py --reports $Reports --folds $Folds

Write-Host "Emit run manifest..." -ForegroundColor Cyan
& $py .\tools\emit_run_manifest.py --reports $Reports

Write-Host "`nArtifacts:" -ForegroundColor Green
Get-ChildItem -File (Join-Path $Reports "wk5_walkforward.parquet"), (Join-Path $Reports "run_manifest.jsonl") `
  -ErrorAction SilentlyContinue | Select-Object Name, Length, LastWriteTime | Format-Table -AutoSize

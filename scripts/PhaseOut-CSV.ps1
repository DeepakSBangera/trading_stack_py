# === scripts\PhaseOut-CSV.ps1 =========================================
[CmdletBinding()]
param(
  [string]$Reports = "reports",
  [switch]$IncludeGovernance  # also convert & delete *_tearsheet.csv, wk*.csv
)

$py = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "Python venv not found at $py" }

Write-Host "Converting CSV → Parquet (and deleting CSVs) under '$Reports' ..." -ForegroundColor Cyan

# Build argument list safely
$argsList = @(".\tools\report_to_parquet.py", "--reports", $Reports)
if ($IncludeGovernance) { $argsList += "--include-governance" }
$argsList += "--delete-csv"

# Run converter
& $py @argsList

# Show remaining CSVs, if any
$left = Get-ChildItem -File (Join-Path $Reports "*.csv") -ErrorAction SilentlyContinue
if ($left) {
  Write-Host "`nCSV files still present (review whether to keep):" -ForegroundColor Yellow
  $left | Select-Object Name, Length, LastWriteTime | Format-Table -AutoSize
} else {
  Write-Host "✓ All target CSVs converted and removed." -ForegroundColor Green
}

[CmdletBinding()]
param(
  [ValidateSet('csv','html')]
  [string]$Format = 'csv',
  [int]$Limit = 2000
)

$ErrorActionPreference = 'Stop'
Set-Location 'F:\Projects\trading_stack_py'

$venvPy = '.\.venv\Scripts\python.exe'
$parquet = '.\reports\factor_exposures.parquet'

if (-not (Test-Path -LiteralPath $venvPy)) { throw 'venv python not found: .\.venv\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $parquet)) { throw 'missing: reports\factor_exposures.parquet' }

# Ensure repo importability (not strictly required here, but harmless)
$repo = (Resolve-Path '.').Path
if ($env:PYTHONPATH) { $env:PYTHONPATH = "$repo;$($env:PYTHONPATH)" } else { $env:PYTHONPATH = $repo }

if ($Format -eq 'csv') {
  $tool = '.\tools\export_parquet_csv.py'
  if (-not (Test-Path -LiteralPath $tool)) { throw 'missing tool: tools\export_parquet_csv.py' }
  & $venvPy $tool $parquet --limit $Limit
  if ($LASTEXITCODE -ne 0) { throw 'CSV export failed' }
  $csv = '.\reports\factor_exposures.csv'
  if (Test-Path -LiteralPath $csv) { Start-Process -FilePath $csv }
}
else {
  $tool = '.\tools\export_parquet_html.py'
  if (-not (Test-Path -LiteralPath $tool)) { throw 'missing tool: tools\export_parquet_html.py' }
  & $venvPy $tool $parquet --limit $Limit --title 'Factor Exposures Preview'
  if ($LASTEXITCODE -ne 0) { throw 'HTML export failed' }
  $html = '.\reports\factor_exposures.html'
  if (Test-Path -LiteralPath $html) { Start-Process -FilePath $html }
}

Write-Host 'Viewer export complete.'

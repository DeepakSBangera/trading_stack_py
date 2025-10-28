[CmdletBinding()]
param(
  [int]$Window = 63,
  [switch]$Open
)

$ErrorActionPreference = 'Stop'

# repo root
Set-Location 'F:\Projects\trading_stack_py'

$venvPy   = '.\.venv\Scripts\python.exe'
$tool     = '.\tools\make_factor_exposures.py'
$templater= '.\tools\write_sector_template.py'
$mapping  = '.\config\sector_mapping.csv'
$outdir   = '.\reports'

if (-not (Test-Path -LiteralPath $venvPy))   { throw 'venv python not found: .\.venv\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $tool))     { throw 'tool not found: .\tools\make_factor_exposures.py' }
if (-not (Test-Path -LiteralPath $templater)){ throw 'templater not found: .\tools\write_sector_template.py' }

# ensure tradingstack is importable
$repo = (Resolve-Path '.').Path
if ($env:PYTHONPATH) { $env:PYTHONPATH = "$repo;$($env:PYTHONPATH)" } else { $env:PYTHONPATH = $repo }

# ensure mapping (deterministic)
if (-not (Test-Path -LiteralPath $mapping)) {
  Write-Host 'Creating sector_mapping.csv template...'
  & $venvPy $templater --weights 'reports/weights_v2_norm.parquet' --out $mapping
  if ($LASTEXITCODE -ne 0) { throw 'templater failed to create sector_mapping.csv' }
  if (-not (Test-Path -LiteralPath $mapping)) { throw 'sector_mapping.csv not created' }
  notepad $mapping | Out-Null
  Write-Host 'Fill the sector column and run again.'
  return
}

# run factor exposures
& $venvPy $tool --window $Window
if ($LASTEXITCODE -ne 0) { throw 'make_factor_exposures failed' }

# tracker updates (best-effort)
$summaryPath = Join-Path -Path $outdir -ChildPath 'factor_exposures_summary.txt'
if (Test-Path -LiteralPath $summaryPath) {
  $summary = Get-Content -Raw -Encoding UTF8 -LiteralPath $summaryPath
  $header  = '## G) Factor Exposures (rolling sector, momentum, quality)'
  $body    = '**Latest snapshot**' + "`r`n`r`n" + '```text' + "`r`n" + $summary + "`r`n" + '```'
  if (Test-Path -LiteralPath .\scripts\Update-TrackerSection.ps1) {
    & .\scripts\Update-TrackerSection.ps1 -Header $header -Body $body 2>$null
  }
  if (Test-Path -LiteralPath .\scripts\Add-TrackerChangelog.ps1) {
    & .\scripts\Add-TrackerChangelog.ps1 -Message 'Session 3: Factor exposures generated'
  }
}

# open outputs if requested
if ($Open) {
  $parquet = Join-Path -Path $outdir -ChildPath 'factor_exposures.parquet'
  $txt     = Join-Path -Path $outdir -ChildPath 'factor_exposures_summary.txt'
  if (Test-Path -LiteralPath $parquet) { Start-Process -FilePath $parquet }
  if (Test-Path -LiteralPath $txt)     { Start-Process -FilePath $txt }
}

Write-Host 'Factor exposures completed.'

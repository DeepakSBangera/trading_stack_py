[CmdletBinding()]
param(
  [string]$Start,
  [string]$UniverseCsv,
  [int]$RollingWindow,
  [switch]$Archive,
  [switch]$Open
)

$ErrorActionPreference = 'Stop'
Set-Location 'F:\Projects\trading_stack_py'

$venvPy = '.\.venv\Scripts\python.exe'
$repo   = (Resolve-Path '.').Path
$env:PYTHONPATH = if ($env:PYTHONPATH) { "$repo;$($env:PYTHONPATH)" } else { $repo }

# ---- load config (JSON from Python) ----
$cfg = $null
if (Test-Path -LiteralPath '.\tools\read_reporting_config.py') {
  $json = & $venvPy .\tools\read_reporting_config.py
  if ($LASTEXITCODE -eq 0 -and $json) {
    $cfg = $json | ConvertFrom-Json
  }
}

# Resolve effective options: CLI overrides config
$effStart   = if ($PSBoundParameters.ContainsKey('Start')) { $Start } elseif ($cfg) { $cfg.start } else { '2025-01-01' }
$effUniCsv  = if ($PSBoundParameters.ContainsKey('UniverseCsv')) { $UniverseCsv } elseif ($cfg) { $cfg.universe_csv } else { 'config/universe.csv' }
$effWindow  = if ($PSBoundParameters.ContainsKey('RollingWindow')) { $RollingWindow } elseif ($cfg) { [int]$cfg.rolling_window } else { 63 }
$effArchive = if ($PSBoundParameters.ContainsKey('Archive')) { $Archive } elseif ($cfg) { [bool]$cfg.archive_after } else { $false }
$effOpen    = if ($PSBoundParameters.ContainsKey('Open')) { $Open } elseif ($cfg) { [bool]$cfg.open_after } else { $false }

# ---- 1) Portfolio ----
if (Test-Path -LiteralPath .\scripts\Report-PortfolioV2.ps1) {
  $ht = @{}
  if ($effStart)     { $ht['Start'] = $effStart }
  if ($effUniCsv)    { $ht['UniverseCsv'] = $effUniCsv }
  & .\scripts\Report-PortfolioV2.ps1 @ht
}

# ---- 2) Rolling metrics ----
if (Test-Path -LiteralPath .\scripts\Make-RollingMetrics.ps1) {
  & .\scripts\Make-RollingMetrics.ps1 -Window $effWindow
} else {
  if (-not (Test-Path -LiteralPath .\tools\make_rolling_metrics.py)) { throw 'missing: tools\make_rolling_metrics.py' }
  & $venvPy .\tools\make_rolling_metrics.py --window $effWindow
  if ($LASTEXITCODE -ne 0) { throw 'make_rolling_metrics failed' }
}

# ---- 3) Factor exposures ----
if (Test-Path -LiteralPath .\scripts\Make-FactorExposures.ps1) {
  & .\scripts\Make-FactorExposures.ps1
} else {
  throw 'missing: scripts\Make-FactorExposures.ps1'
}

# ---- 4) Tearsheet v2 ----
if (-not (Test-Path -LiteralPath .\scripts\Make-TearsheetV2.ps1)) { throw 'missing: scripts\Make-TearsheetV2.ps1' }
& .\scripts\Make-TearsheetV2.ps1

# ---- 5) Report index ----
if (-not (Test-Path -LiteralPath .\tools\make_report_index.py)) { throw 'missing: tools\make_report_index.py' }
& $venvPy .\tools\make_report_index.py
if ($LASTEXITCODE -ne 0) { throw 'make_report_index failed' }

# ---- 6) Archive (optional) ----
if ($effArchive -and (Test-Path -LiteralPath .\scripts\Archive-Reports.ps1)) {
  & .\scripts\Archive-Reports.ps1
}

# ---- 7) Tracker (best-effort) ----
if (Test-Path -LiteralPath .\scripts\Add-TrackerChangelog.ps1) {
  & .\scripts\Add-TrackerChangelog.ps1 -Message "Session 5: Daily report (Start=$effStart, Window=$effWindow, Universe=$effUniCsv)"
}
if (Test-Path -LiteralPath .\scripts\Update-TrackerSection.ps1) {
  $body = "Daily report index: reports\index_daily.html`nRollingWindow: $effWindow`nStart: $effStart`nUniverseCsv: $effUniCsv"
  & .\scripts\Update-TrackerSection.ps1 -Header '## I) Reporting Layer (Session 5)' -Body $body 2>$null
}

# ---- 8) Open (optional) ----
if ($effOpen) {
  $ix = '.\reports\index_daily.html'
  if (Test-Path -LiteralPath $ix) { Start-Process -FilePath $ix }
}

Write-Host 'Daily report completed.'

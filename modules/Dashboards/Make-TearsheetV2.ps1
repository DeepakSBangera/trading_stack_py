[CmdletBinding()]
param(
  [switch]$Open
)

$ErrorActionPreference = 'Stop'
Set-Location 'F:\Projects\trading_stack_py'

$venvPy = '.\.venv\Scripts\python.exe'
$tool   = '.\tools\plot_tearsheet_v2.py'
$outdir = '.\reports'
$png    = Join-Path $outdir 'tearsheet_v2.png'
$html   = Join-Path $outdir 'tearsheet_v2.html'

if (-not (Test-Path -LiteralPath $venvPy)) { throw 'venv python not found: .\.venv\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $tool))   { throw 'tool not found: .\tools\plot_tearsheet_v2.py' }

# Ensure tradingstack package importable
$repo = (Resolve-Path '.').Path
if ($env:PYTHONPATH) { $env:PYTHONPATH = "$repo;$($env:PYTHONPATH)" } else { $env:PYTHONPATH = $repo }

# Run
& $venvPy $tool
if ($LASTEXITCODE -ne 0) { throw 'plot_tearsheet_v2 failed' }

# Update tracker (best-effort)
if (Test-Path -LiteralPath .\scripts\Update-TrackerSection.ps1) {
  $body = '**Latest snapshot (Tearsheet v2)**' + "`r`n`r`n" + 'File: reports\tearsheet_v2.png' + "`r`n" + 'HTML: reports\tearsheet_v2.html'
  & .\scripts\Update-TrackerSection.ps1 -Header '## H) Tearsheet v2 Snapshot' -Body $body 2>$null
}
if (Test-Path -LiteralPath .\scripts\Add-TrackerChangelog.ps1) {
  & .\scripts\Add-TrackerChangelog.ps1 -Message 'Session 4: Tearsheet v2 generated'
}

# Open if requested
if ($Open) {
  if (Test-Path -LiteralPath $html) { Start-Process -FilePath $html }
  elseif (Test-Path -LiteralPath $png) { Start-Process -FilePath $png }
}

Write-Host 'Tearsheet v2 completed.'

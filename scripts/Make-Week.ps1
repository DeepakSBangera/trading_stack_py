[CmdletBinding()]
param(
  [Parameter(Mandatory=$true)][string] $Week,
  [string] $RepoRoot = (Resolve-Path ".").Path
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$RepoRoot = (Resolve-Path $RepoRoot).Path
$cfg = Join-Path $RepoRoot 'config\run.yaml'
if (-not (Test-Path $cfg)) { throw "Missing config: $cfg" }

$raw = Get-Content -LiteralPath $cfg -Raw
$start = if ($raw -match 'start_date:\s*"([^"]+)"') { $matches[1] } else { '' }
$reportsDir = if ($raw -match 'reports_dir:\s*"([^"]+)"') { $matches[1] } else { 'reports' }
$reports = Join-Path $RepoRoot $reportsDir
if (-not (Test-Path $reports)) { New-Item -ItemType Directory -Force $reports | Out-Null }

switch -Regex ($Week.ToUpper()) {
  '^W3$'  { if (Test-Path .\scripts\Make-Turnover.ps1) { pwsh .\scripts\Make-Turnover.ps1 }
            if (Test-Path .\scripts\Make-LiquidityScreens.ps1) { pwsh .\scripts\Make-LiquidityScreens.ps1 } }
  '^W4$'  { if (Test-Path .\scripts\Make-VolTargetStops.ps1) { pwsh .\scripts\Make-VolTargetStops.ps1 -Start $start } }
  '^W5$'  { if (Test-Path .\scripts\Make-WalkForwardDSR.ps1) { pwsh .\scripts\Make-WalkForwardDSR.ps1 -Start $start } }
  '^W6$'  { if (Test-Path .\scripts\Report-PortfolioV2.ps1) { pwsh .\scripts\Report-PortfolioV2.ps1 -Start $start }
            if (Test-Path .\scripts\Make-FactorExposures.ps1) { pwsh .\scripts\Make-FactorExposures.ps1 }
            if (Test-Path .\scripts\Make-CapacityCurve.ps1) { pwsh .\scripts\Make-CapacityCurve.ps1 } }
  default { throw "Unknown or not-yet-wired week: $Week" }
}

if (Test-Path .\tools\run_manifest.py) {
  .\.venv\Scripts\python.exe .\tools\run_manifest.py --week $Week --config $cfg --reports $reports `
    --git-sha "$(git rev-parse HEAD)" --when "$(Get-Date -Format s)" | Out-Null
}

Write-Host "[OK] Week $Week completed." -ForegroundColor Green

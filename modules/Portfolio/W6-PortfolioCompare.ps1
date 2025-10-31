# tools/W6-PortfolioCompare.ps1
[CmdletBinding()]
param(
  [Parameter(Mandatory=$true )][string]$Start,
  [Parameter(Mandatory=$false)][string]$UniverseCsv = ".\config\my_universe.csv",
  [Parameter(Mandatory=$false)][string]$UniverseDir = ".\data_synth\prices",
  [int]$Lookback = 252,
  [ValidateSet("ME","WE","QE")][string]$Rebalance = "ME",
  [double]$CostBps = 10,
  [string]$OutDir = "reports"
)

$ErrorActionPreference = "Stop"
$py = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "Python venv not found at $py" }
if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Force -Path $OutDir | Out-Null }

$argsList = @(".\tools\w6_portfolio_compare.py",
              "--universe-csv", $UniverseCsv,
              "--prices-root",  $UniverseDir,
              "--start",        $Start,
              "--lookback",     $Lookback,
              "--rebalance",    $Rebalance,
              "--cost-bps",     $CostBps,
              "--outdir",       $OutDir)

Write-Host "Running: w6_portfolio_compare.py"
& $py @argsList
if ($LASTEXITCODE -ne 0) { throw "W6 compare exited with code $LASTEXITCODE" }

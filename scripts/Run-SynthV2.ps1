[CmdletBinding()]
param(
  [string]$PricesRoot = "data_synth\prices",
  [string]$Start = "2015-01-01",
  [int]$Lookback = 126,
  [int]$TopN = 4,
  [ValidateSet("ME","WE","QE")][string]$Rebalance = "ME",
  [double]$CostBps = 10,
  [string]$OutDir = "reports"
)
$py = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "Python venv not found at $py" }

Write-Host "Running synthetic V2..." -ForegroundColor Cyan
& $py .\tools\report_portfolio_synth_v2.py `
  --universe-csv (.\tools\Report-PortfolioV2.ps1 -UniverseDir $PricesRoot -Start $Start -Lookback $Lookback -TopN $TopN -Rebalance $Rebalance -CostBps $CostBps -OutDir $OutDir)  # not actually used; wrapper already runs it

param(
  [string]$UniverseDir = "data_synth\prices",
  [string]$Start = "2015-01-01",
  [int]$Lookback = 126,
  [int]$TopN = 4,
  [string]$Rebalance = "ME",
  [double]$CostBps = 10,
  [string]$OutDir = "reports"
)
.\tools\Report-PortfolioV2.ps1 -UniverseDir $UniverseDir -Start $Start -Lookback $Lookback -TopN $TopN -Rebalance $Rebalance -CostBps $CostBps -OutDir $OutDir

# find the newest portfolioV2_* CSV and its weights
$latest = Get-ChildItem -File $OutDir\portfolioV2_*.csv | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($null -eq $latest) { throw "No portfolioV2_*.csv found in $OutDir" }
$weights = [System.IO.Path]::Combine($OutDir, ([System.IO.Path]::GetFileNameWithoutExtension($latest.FullName) + "_weights.csv"))
& .\.venv\Scripts\python.exe .\tools\make_tearsheet.py --equity-csv $latest.FullName --weights-csv $weights

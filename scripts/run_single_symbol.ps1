param(
  [Parameter(Mandatory=$true)][string]$Ticker,
  [string]$Start="2018-01-01",
  [double]$CostBps=12
)
python scripts/runner_cli.py --ticker $Ticker --start $Start --cost_bps $CostBps

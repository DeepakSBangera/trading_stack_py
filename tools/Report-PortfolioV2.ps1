<#  Report-PortfolioV2.ps1
    Runs modules\Portfolio\report_portfolio_v2.py using the repo venv.

    Examples:
      .\tools\Report-PortfolioV2.ps1 -OutDir ".\reports" -Start "2024-01-01" -UniverseCsv ".\data\universe\nifty50.csv" -Verbose
      .\tools\Report-PortfolioV2.ps1 -OutDir ".\reports" -Start "2024-01-01" -Tickers "RELIANCE.NS,TCS.NS,INFY.NS" -Verbose
      .\tools\Report-PortfolioV2.ps1 -OutDir ".\reports" -DryRun -Verbose
#>

[CmdletBinding()]
param(
  [string]$OutDir = "$PSScriptRoot\..\reports",
  [switch]$DryRun,

  # Strategy knobs (defaults are sane for a quick run)
  [string]$Start = "2024-01-01",
  [int]$Lookback = 252,
  [int]$TopN = 20,
  [int]$MaxHoldings = 20,
  [double]$WeightCap = 0.15,
  [double]$MinWeight = 0.0,
  [double]$CashBuffer = 0.0,
  [ValidateSet("ME","WE","QE")][string]$Rebalance = "WE",
  [int]$CostBps = 5,
  [int]$MgmtFeeBps = 0,
  [double]$VolTarget = 0.0,
  [double]$TurnoverBand = 0.0,
  [string]$Benchmark = "^NSEI",

  # Universe (provide either of these; if neither, auto-discover)
  [string]$Tickers = "",
  [string]$UniverseCsv = ""
)

$ErrorActionPreference = "Stop"

function Resolve-RepoRoot {
  return (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

function Resolve-Python {
  param([string]$RepoRoot)
  $candidates = @(
    (Join-Path $RepoRoot ".venv\Scripts\python.exe"),
    (Join-Path $RepoRoot "venv\Scripts\python.exe"),
    (Join-Path $RepoRoot "Scripts\python.exe")
  )
  foreach($p in $candidates){ if(Test-Path $p){ return $p } }
  $cmd = Get-Command python -ErrorAction SilentlyContinue
  if($cmd){ return $cmd.Source }
  throw "Python not found. Expected venv at '$($candidates[0])'. Activate your venv or ensure 'python' is on PATH."
}

function Resolve-ReportScript {
  param([string]$RepoRoot)
  $p = Join-Path $RepoRoot "modules\Portfolio\report_portfolio_v2.py"
  if(Test-Path $p){ return $p }
  throw "report_portfolio_v2.py not found at: $p"
}

# -------- main --------
$repoRoot  = Resolve-RepoRoot
$pythonExe = Resolve-Python -RepoRoot $repoRoot
$reportPy  = Resolve-ReportScript -RepoRoot $repoRoot

# Ensure OutDir exists and is absolute
$OutDir = (Resolve-Path (New-Item -ItemType Directory -Force -Path $OutDir)).Path

# If neither universe option provided, auto-discover (limit & equities only to avoid long filenames)
if (-not $Tickers -and -not $UniverseCsv) {
  $csvCandidate = Join-Path $repoRoot "data\universe\nifty50.csv"
  if (Test-Path $csvCandidate) {
    Write-Verbose "Auto-using universe CSV: $csvCandidate"
    $UniverseCsv = $csvCandidate
  } else {
    $pricesDir = Join-Path $repoRoot "data\prices"
    if (Test-Path $pricesDir) {
      $names = Get-ChildItem $pricesDir -Filter *.NS.parquet -File |
               Select-Object -First 15 |
               ForEach-Object { $_.BaseName }
      if ($names.Count -gt 0) {
        $Tickers = ($names -join ',')
        Write-Verbose ("Auto-using equities (.NS) from parquet files ({0} found; first 15 used)" -f $names.Count)
      }
    }
  }
}

# Build python args
$argsList = @(
  $reportPy,
  "--outdir", $OutDir,
  "--start", $Start,
  "--lookback", $Lookback,
  "--top-n", $TopN,
  "--max-holdings", $MaxHoldings,
  "--weight-cap", $WeightCap,
  "--min-weight", $MinWeight,
  "--cash-buffer", $CashBuffer,
  "--rebalance", $Rebalance,
  "--cost-bps", $CostBps,
  "--mgmt-fee-bps", $MgmtFeeBps,
  "--vol-target", $VolTarget,
  "--turnover-band", $TurnoverBand,
  "--benchmark", $Benchmark
)

if ($UniverseCsv) { $argsList += @("--universe-csv", $UniverseCsv) }
elseif ($Tickers) { $argsList += @("--tickers", $Tickers) }

Write-Verbose ("RepoRoot : {0}" -f $repoRoot)
Write-Verbose ("Python   : {0}" -f $pythonExe)
Write-Verbose ("Script   : {0}" -f $reportPy)
Write-Verbose ("OutDir   : {0}" -f $OutDir)
Write-Verbose ("Start    : {0}" -f $Start)

Write-Host "Running report generator..." -ForegroundColor Cyan

if ($DryRun) {
  Write-Host "DRY RUN - would execute:" -ForegroundColor Yellow
  Write-Host ("`"{0}`" {1}" -f $pythonExe, ($argsList -join ' ')) -ForegroundColor Yellow
  exit 0
}

& $pythonExe @argsList
$code = $LASTEXITCODE
if ($code -ne 0) { throw "report_portfolio_v2.py exited with code $code" }

Write-Host ("Report generation completed. Files should be in: {0}" -f $OutDir) -ForegroundColor Green

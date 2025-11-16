param(
  [string]$Universe = "data\csv\universe_small.csv",
  [string]$Prices   = "data\prices",
  [string]$Start    = "2025-01-01",
  [string]$OutDir   = "reports",
  [int]   $Lookback = 252,
  [ValidateSet("ME","WE","QE")] [string]$Rebalance = "ME",
  [int]   $CostBps  = 5
)

$ErrorActionPreference = 'Stop'
$py = ".\.venv\Scripts\python.exe"
$w6 = "modules\Portfolio\w6_portfolio_compare.py"

if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Force $OutDir | Out-Null }

# If universe file is missing, synthesize a tiny one from whatever prices exist
if (-not (Test-Path $Universe)) {
  $priceDir = Resolve-Path $Prices -ErrorAction Stop
  $tickers = Get-ChildItem -LiteralPath $priceDir -Filter "*.parquet" -File -ErrorAction SilentlyContinue |
             Select-Object -First 15 | ForEach-Object { [System.IO.Path]::GetFileNameWithoutExtension($_.Name) }
  if (-not $tickers) { throw "No price files found in $Prices" }
  "ticker" | Set-Content $Universe -Encoding UTF8
  $tickers | ForEach-Object { Add-Content $Universe $_ }
}

& $py $w6 `
  --universe-csv $Universe `
  --prices-root  $Prices `
  --start        $Start `
  --lookback     $Lookback `
  --rebalance    $Rebalance `
  --cost-bps     $CostBps `
  --outdir       $OutDir

if ($LASTEXITCODE) { throw "W6 failed" } else { Write-Host "[OK] W6 artifacts under $OutDir" -ForegroundColor Green }

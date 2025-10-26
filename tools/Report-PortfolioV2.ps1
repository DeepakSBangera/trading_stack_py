[CmdletBinding()]
param(
  [Parameter(Mandatory=$false)][string]$UniverseDir = "",
  [Parameter(Mandatory=$false)][string]$UniverseCsv = "",
  [Parameter(Mandatory=$true )][string]$Start,
  [int]$Lookback = 126,
  [int]$TopN = 4,
  [ValidateSet("ME","WE","QE")][string]$Rebalance = "ME",
  [double]$CostBps = 10,
  [string]$OutDir = "reports"
)

$ErrorActionPreference = "Stop"
$py = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "Python venv not found at $py" }
if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Force -Path $OutDir | Out-Null }

function Build-UniverseCsv {
  param([string]$DirPath)
  if (-not (Test-Path $DirPath)) { throw "UniverseDir not found: $DirPath" }

  $files = @(Get-ChildItem -Recurse -File -ErrorAction SilentlyContinue `
    -Path $DirPath | Where-Object { $_.Extension -match '\.parquet$' -and $_.FullName -notmatch '_fundamentals\.parquet$' })
  if ($files.Count -eq 0) { throw "No *.parquet found in $DirPath" }

  # RELIANCE_NS -> RELIANCE.NS ; LT -> LT (unchanged)
  $syms = $files | ForEach-Object {
    $b = $_.BaseName -replace '_fundamentals$', ''
    if ($b -match '^(.*)_NS$') { "$($matches[1]).NS" }
    elseif ($b -match '^(.*)_NSE$') { "$($matches[1]).NS" }
    else { $b -replace '_', '.' }
  } | Sort-Object -Unique

  if ($syms.Count -eq 0) { throw "Could not derive any symbols from $DirPath" }

  $tmp = Join-Path $env:TEMP ("universe_{0}.csv" -f [guid]::NewGuid())
  "ticker" | Set-Content -Path $tmp -Encoding UTF8
  $syms   | Add-Content -Path $tmp -Encoding UTF8
  Write-Host ("Built temp universe CSV ({0} symbols): {1}" -f $syms.Count, $tmp)
  return $tmp
}

$uCsv = $UniverseCsv
if ([string]::IsNullOrWhiteSpace($uCsv) -and -not [string]::IsNullOrWhiteSpace($UniverseDir)) {
  $uCsv = Build-UniverseCsv -DirPath $UniverseDir
}
if ([string]::IsNullOrWhiteSpace($uCsv)) { throw "Provide -UniverseDir or -UniverseCsv (header: 'ticker')." }
if (-not (Test-Path $uCsv)) { throw "Universe CSV not found: $uCsv" }

# Instead of calling Yahoo v2, call our synthetic runner (next section)
$argsList = @(".\tools\report_portfolio_synth_v2.py",
              "--universe-csv", $uCsv,
              "--prices-root", $UniverseDir,
              "--start", $Start,
              "--lookback", $Lookback,
              "--top-n", $TopN,
              "--rebalance", $Rebalance,
              "--cost-bps", $CostBps,
              "--outdir", $OutDir)

Write-Host "Running: report_portfolio_synth_v2.py"
& $py @argsList
if ($LASTEXITCODE -ne 0) { throw "Backtest exited with code $LASTEXITCODE" }

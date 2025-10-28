# Report-PortfolioV2.ps1 — run portfolio v2 report with optional date/ticker inputs
# Usage examples:
#   pwsh .\scripts\Report-PortfolioV2.ps1
#   pwsh .\scripts\Report-PortfolioV2.ps1 -Start 2025-01-01
#   pwsh .\scripts\Report-PortfolioV2.ps1 -UniverseCsv .\config\universe.csv
#   pwsh .\scripts\Report-PortfolioV2.ps1 -Tickers "AXISBANK.NS,HDFCBANK.NS,INFY.NS"
#   pwsh .\scripts\Report-PortfolioV2.ps1 -Open

[CmdletBinding()]
Param(
  [string]$Start,
  [string]$End,                 # kept for future compatibility; runner may ignore
  [string]$UniverseCsv,
  [string]$Tickers,
  [switch]$Open
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Repo root
Set-Location F:\Projects\trading_stack_py

$venvPy = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) { throw "Virtual env python not found at $venvPy" }

$equityParquet = ".\reports\portfolio_v2.parquet"
if (-not (Test-Path $equityParquet)) { throw "Missing $equityParquet. Generate it first." }

# Choose the report runner (prefer non-synth if both exist)
$runner =
  if (Test-Path ".\tools\report_portfolio_v2.py") { ".\tools\report_portfolio_v2.py" }
  elseif (Test-Path ".\tools\report_portfolio_synth_v2.py") { ".\tools\report_portfolio_synth_v2.py" }
  else { $null }

if (-not $runner) {
  throw "No report runner found (tools\report_portfolio_v2.py or tools\report_portfolio_synth_v2.py)."
}

function Get-DateBoundsFromParquet([string]$path) {
  $py = @"
from pathlib import Path
import pandas as pd
p = Path(r'$path')
df = pd.read_parquet(p)
if 'date' in df.columns:
    s = pd.to_datetime(df['date'], utc=True, errors='coerce')
else:
    s = pd.to_datetime(df.index, utc=True, errors='coerce')
try:
    s = s.dt.tz_convert(None)
except Exception:
    pass
try:
    s = s.dt.tz_localize(None)
except Exception:
    pass
s = s.dropna()
print(s.min().strftime('%Y-%m-%d') if len(s) else '')
print(s.max().strftime('%Y-%m-%d') if len(s) else '')
"@
  $tmp = Join-Path $env:TEMP ("get_bounds_{0}.py" -f [guid]::NewGuid().ToString("N"))
  Set-Content -LiteralPath $tmp -Encoding UTF8 -Value $py
  try {
    $out = & $venvPy $tmp 2>&1
    $lines = $out -split "`r?`n"
    return @{ Min = $lines[0]; Max = $lines[1] }
  } finally {
    Remove-Item $tmp -ErrorAction Ignore
  }
}

function Infer-Tickers {
  # Prefer weights_v2_norm.parquet → ticker column
  if (Test-Path ".\reports\weights_v2_norm.parquet") {
    $py = @"
import pandas as pd
df = pd.read_parquet(r'.\reports\weights_v2_norm.parquet')
t = pd.Series(df.get('ticker', pd.Series(dtype=object))).dropna().astype(str).unique().tolist()
print(','.join(t))
"@
    $tmp = Join-Path $env:TEMP ("get_tickers_{0}.py" -f [guid]::NewGuid().ToString("N"))
    Set-Content -LiteralPath $tmp -Encoding UTF8 -Value $py
    try {
      $out = (& $venvPy $tmp).Trim()
      if ($out) { return $out }
    } finally {
      Remove-Item $tmp -ErrorAction Ignore
    }
  }
  # Fallback: columns of attribution_ticker.parquet
  if (Test-Path ".\reports\attribution_ticker.parquet") {
    $py = @"
import pandas as pd
df = pd.read_parquet(r'.\reports\attribution_ticker.parquet')
cols = [c for c in df.columns if c.lower() != 'date']
print(','.join(cols))
"@
    $tmp = Join-Path $env:TEMP ("get_tickers_{0}.py" -f [guid]::NewGuid().ToString("N"))
    Set-Content -LiteralPath $tmp -Encoding UTF8 -Value $py
    try {
      $out = (& $venvPy $tmp).Trim()
      if ($out) { return $out }
    } finally {
      Remove-Item $tmp -ErrorAction Ignore
    }
  }
  return ""
}

function Ensure-UniverseCsv([string]$path) {
  # Creates universe CSV if missing, prefilled with inferred tickers; opens Notepad.
  $full = Resolve-Path -Path $path -ErrorAction Ignore
  if (-not $full) {
    # ensure folder exists
    $dir = Split-Path -Parent $path
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir | Out-Null }

    $tickersCsv = Infer-Tickers
    if (-not $tickersCsv) {
      # fallback: 5 common tickers as a template
      $tickersCsv = "AXISBANK.NS,HDFCBANK.NS,INFY.NS,RELIANCE.NS,TCS.NS"
    }
    $lines = @("ticker") + ($tickersCsv -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ })
    Set-Content -LiteralPath $path -Encoding UTF8 -Value ($lines -join "`r`n")
    Write-Host "[Created] $path (prefilled $(($lines.Count - 1)) tickers). Review & save."
    notepad $path | Out-Null
  } else {
    # path exists; nothing to do
    $path = $full.Path
  }
  return (Resolve-Path -Path $path).Path
}

# Fill Start/End if missing
$bounds = Get-DateBoundsFromParquet -path $equityParquet
if ([string]::IsNullOrWhiteSpace($Start)) { $Start = $bounds.Min }
if ([string]::IsNullOrWhiteSpace($End))   { $End   = $bounds.Max }  # may be unused by runner

if (-not [string]::IsNullOrWhiteSpace($Start) -and $Start -notmatch '^\d{4}-\d{2}-\d{2}$') { throw "Start must be YYYY-MM-DD. Got '$Start'." }
if (-not [string]::IsNullOrWhiteSpace($End)   -and $End   -notmatch '^\d{4}-\d{2}-\d{2}$') { throw "End must be YYYY-MM-DD. Got '$End'." }

# Resolve universe
$tickersCsv = $null
$universePath = $null

if (-not [string]::IsNullOrWhiteSpace($Tickers)) {
  $tickersCsv = ($Tickers -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ }) -join ","
} else {
  if ([string]::IsNullOrWhiteSpace($UniverseCsv)) {
    # default to config\universe.csv; create if missing
    $UniverseCsv = ".\config\universe.csv"
  }
  if (-not (Test-Path $UniverseCsv)) {
    $universePath = Ensure-UniverseCsv -path $UniverseCsv
  } else {
    $universePath = (Resolve-Path -Path $UniverseCsv).Path
  }
}

# Build args (runner likely needs --start; many do not accept --end)
$argList = @("--start", $Start)
if ($universePath) {
  $argList += @("--universe-csv", $universePath)
} elseif ($tickersCsv) {
  # limit to 100 to keep CLI manageable
  $tickersCsv = ($tickersCsv -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ } | Select-Object -First 100) -join ","
  $argList += @("--tickers", $tickersCsv)
} else {
  throw "No universe available. Provide -UniverseCsv or -Tickers, or ensure weights_v2_norm.parquet / attribution_ticker.parquet exists."
}

Write-Host "[Run] $runner $($argList -join ' ')"
& $venvPy $runner @argList
if ($LASTEXITCODE -ne 0) {
  Write-Warning "Runner exited with code $LASTEXITCODE. Check script usage or inputs."
}

# Open outputs if requested
$reportTxt = ".\reports\portfolio_v2_summary.txt"
$reportPng = ".\reports\portfolio_v2.png"
if ($Open) {
  if (Test-Path $reportTxt) { Start-Process $reportTxt }
  if (Test-Path $reportPng) { Start-Process $reportPng }
  if (Test-Path $equityParquet) { Start-Process $equityParquet }
}

# Tickers count display
$tCount = 0
if ($tickersCsv) { $tCount = ($tickersCsv -split "," | Where-Object { $_ } | Measure-Object).Count }
elseif ($universePath -and (Test-Path $universePath)) {
  try {
    $csv = Import-Csv -LiteralPath $universePath
    if ($csv) { $tCount = ($csv | ForEach-Object { $_.ticker } | Where-Object { $_ } | Measure-Object).Count }
  } catch { }
}

Write-Host "[OK] Done. Start=$Start UniverseCsv=$UniverseCsv TickersCount=$tCount"

Param(
  [string]$Root = ".",
  [string[]]$Modules = @(
    "tradingstack\metrics\sharpe.py",
    "tradingstack\metrics\sortino.py",
    "tradingstack\metrics\drawdown.py",
    "tradingstack\metrics\calmar.py",
    "tradingstack\metrics\omega.py"
  )
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$rootPath = (Resolve-Path $Root).Path
$targets = @{}
foreach ($m in $Modules) {
  $p = Join-Path $rootPath $m
  if (Test-Path $p) { $targets[$m] = (Resolve-Path $p).Path }
}

# Simple function scanner (top-level def names)
function Get-TopLevelDefs([string]$filePath) {
  $text = Get-Content -LiteralPath $filePath -Raw -Encoding UTF8
  # match lines like: def name(...
  $regex = '^[ \t]*def[ \t]+([A-Za-z_]\w*)\s*\('
  $matches = [System.Text.RegularExpressions.Regex]::Matches($text, $regex, 'Multiline')
  $fn = @()
  foreach ($m in $matches) { $fn += $m.Groups[1].Value }
  ,$fn
}

# Canonical export names we want
$canonical = @{
  "tradingstack.metrics.sharpe"   = "sharpe_ratio"
  "tradingstack.metrics.sortino"  = "sortino_ratio"
  "tradingstack.metrics.drawdown" = "max_drawdown"
  "tradingstack.metrics.calmar"   = "calmar_ratio"
  "tradingstack.metrics.omega"    = "omega_ratio"
}

# Build mapping body
$lines = @()
foreach ($rel in $targets.Keys | Sort-Object) {
  $file = $targets[$rel]
  $funcs = Get-TopLevelDefs $file
  $mod = $rel -replace "\\", "/"
  $pkg = ($mod -replace "\.py$","") -replace "/", "."
  $want = $canonical[$pkg]
  $actual = $null

  # heuristic: pick best candidate by name similarity
  $cands = @($funcs | Sort-Object)
  if ($cands -contains $want) { $actual = $want }
  elseif ($cands -contains ($want -replace "_ratio$","")) { $actual = ($want -replace "_ratio$","") }
  elseif ($cands -match "calc_") { $actual = ($cands | Where-Object { $_ -like "calc_*" } | Select-Object -First 1) }
  elseif ($cands.Count -gt 0) { $actual = $cands[0] }

  if (-not $actual) { $actual = "TBD" }
  $lines += ("- `{0}`: (actual: `{1}`) → **`{2}`**" -f $pkg, $actual, $want)
}

$body = ($lines -join "`r`n")

# Update tracker Section I via our updater
& .\scripts\Update-TrackerSection.ps1 `
  -Header "## I) API Compatibility Mapping (actual → canonical)" `
  -Body $body

Write-Host "[OK] Metrics mapping updated." -ForegroundColor Green

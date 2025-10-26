param(
  [string]$ArchiveDir = ".\reports\archive",
  [string]$OutDir     = ".\reports",
  [string]$PythonPath = ".\.venv\Scripts\python.exe"
)
$ErrorActionPreference = "Stop"

if (-not (Test-Path $PythonPath)) { throw "Python venv not found at $PythonPath" }
if (-not (Test-Path $ArchiveDir)) { throw "Archive dir not found: $ArchiveDir" }
if (-not (Test-Path $OutDir))     { New-Item -ItemType Directory -Force -Path $OutDir | Out-Null }

$checker  = ".\tools\parquet_has_date.py"
$compare  = ".\tools\compare_runs.py"
$fallback = ".\tools\compare_runs_fallback.py"
if (-not (Test-Path $checker)) { throw "Missing checker: $checker" }

function Has-Date([string]$Path) {
  & $PythonPath $checker $Path 2>$null | Out-String | Select-Object -First 1 | ForEach-Object { $_.Trim() } | Where-Object { $_ -match '^OK' } | ForEach-Object { $true }
  if ($LASTEXITCODE -eq 0 -and $?) { return $true } else { return $false }
}

# 1) Gather recent portfolio snapshots
$cands = Get-ChildItem $ArchiveDir -Filter "portfolio_v2_*.parquet" -File -ErrorAction SilentlyContinue |
         Sort-Object LastWriteTime -Descending

# 2) Keep only those with a 'date' column
$valid = @()
foreach ($f in $cands) {
  if (Has-Date $f.FullName) {
    $valid += $f
    if ($valid.Count -ge 2) { break }
  }
}

# 3) If we only found one valid archive, try to use current reports\portfolio_v2.parquet as B
$current = ".\reports\portfolio_v2.parquet"
if ($valid.Count -eq 1 -and (Test-Path $current) -and (Has-Date $current)) {
  $a = $valid[0].FullName
  $b = (Resolve-Path $current).Path
} elseif ($valid.Count -ge 2) {
  $a = $valid[1].FullName  # older
  $b = $valid[0].FullName  # newer
} else {
  Write-Host "Need two valid inputs (Parquets with 'date')." -ForegroundColor Yellow
  Write-Host "Do this: 1) run the pipeline, 2) run Archive-Reports.ps1 twice, then re-run this script." -ForegroundColor Yellow
  exit 2
}

Write-Host "Compare →" -ForegroundColor Cyan
Write-Host "A (older): $a"
Write-Host "B (newer): $b"

function Run-Compare {
  param([string]$cmd, [string]$APath, [string]$BPath, [string]$Out)
  & $PythonPath $cmd --a "$APath" --b "$BPath" --out "$Out"
  return ($LASTEXITCODE -eq 0)
}

# 4) Prefer main comparator, fallback if needed
$ran = $false
if (Test-Path $compare) {
  if (Run-Compare -cmd $compare -APath $a -BPath $b -Out $OutDir) { $ran = $true }
  else {
    Write-Warning "compare_runs.py failed ($LASTEXITCODE). Trying fallback…"
  }
}
if (-not $ran -and (Test-Path $fallback)) {
  if (Run-Compare -cmd $fallback -APath $a -BPath $b -Out $OutDir) { $ran = $true }
  else { Write-Error "fallback comparator failed ($LASTEXITCODE)"; exit 3 }
}
if (-not $ran) {
  Write-Error "No comparator found: expected $compare or $fallback"; exit 4
}

Write-Host "Compare written to $OutDir (compare_runs.*)" -ForegroundColor Green

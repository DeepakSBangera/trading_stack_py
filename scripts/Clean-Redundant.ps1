[CmdletBinding()]
param(
  [string] $RepoRoot = (Resolve-Path ".").Path,
  [switch] $Aggressive
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$RepoRoot = (Resolve-Path $RepoRoot).Path

$killDirs = @('__pycache__','.ruff_cache','.pytest_cache','.mypy_cache','.ipynb_checkpoints')

foreach ($k in $killDirs) {
  $hits = Get-ChildItem -LiteralPath $RepoRoot -Recurse -Force -Directory -ErrorAction SilentlyContinue `
    | Where-Object { $_.Name -eq $k }
  foreach ($h in $hits) { try { Remove-Item -Recurse -Force -LiteralPath $h.FullName } catch {} }
}

$dupScopes = @('reports\_tmp','reports\_cache')
foreach ($scope in $dupScopes) {
  $p = Join-Path $RepoRoot $scope
  if (-not (Test-Path $p)) { continue }
  $byHash = @{}
  $files = Get-ChildItem -LiteralPath $p -Recurse -Force -File -ErrorAction SilentlyContinue
  foreach ($f in $files) {
    $h = (Get-FileHash -Algorithm SHA256 -LiteralPath $f.FullName).Hash
    if ($byHash.ContainsKey($h)) {
      Remove-Item -LiteralPath $f.FullName -Force -ErrorAction SilentlyContinue
    } else { $byHash[$h] = $f.FullName }
  }
}

$cfg = Join-Path $RepoRoot 'config\run.yaml'
$keepDays = 90
if (Test-Path $cfg) {
  $raw = Get-Content -LiteralPath $cfg -Raw
  if ($raw -match 'keep_reports_days:\s*(\d+)') { $keepDays = [int]$matches[1] }
}
if ($Aggressive) {
  $reports = Join-Path $RepoRoot 'reports'
  if (Test-Path $reports) {
    $cut = (Get-Date).AddDays(-$keepDays)
    $files = Get-ChildItem -LiteralPath $reports -Recurse -Force -File -ErrorAction SilentlyContinue `
      | Where-Object { $_.LastWriteTime -lt $cut -and ($_.Name -notmatch '^wk\d+_') }
    foreach ($f in $files) { try { Remove-Item -LiteralPath $f.FullName -Force } catch {} }
  }
}

Write-Host "[OK] Redundant cleanup complete." -ForegroundColor Green

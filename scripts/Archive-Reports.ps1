param(
  [string]$ReportsDir = ".\reports",
  [string[]]$Names = @(
    "portfolio_v2.parquet",
    "wk5_walkforward.parquet",
    "weights_v2.parquet"
  )
)
$ErrorActionPreference = "Stop"

if (-not (Test-Path $ReportsDir)) { throw "Reports dir not found: $ReportsDir" }

$arch = Join-Path $ReportsDir "archive"
if (-not (Test-Path $arch)) { New-Item -ItemType Directory -Force -Path $arch | Out-Null }

# Build the candidate list FIRST (no pipelines with foreach)
$paths = @()
foreach ($n in $Names) {
  $paths += (Join-Path $ReportsDir $n)
}

# Now filter with Where-Object
$targets = $paths | Where-Object { Test-Path $_ }

if (-not $targets -or $targets.Count -eq 0) {
  Write-Warning "No report files found to archive in $ReportsDir"
  exit 0
}

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
foreach ($src in $targets) {
  $base = [System.IO.Path]::GetFileNameWithoutExtension($src)
  $ext  = [System.IO.Path]::GetExtension($src)
  $dst  = Join-Path $arch ("{0}_{1}{2}" -f $base,$stamp,$ext)
  Copy-Item -LiteralPath $src -Destination $dst -Force
  Write-Host "Archived $([System.IO.Path]::GetFileName($src)) -> $dst"
}

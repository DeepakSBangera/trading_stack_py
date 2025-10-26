[CmdletBinding()]
param([int]$KeepBases=5, [int]$KeepTears=10, [string]$ReportsDir=".\reports")

$ErrorActionPreference="Stop"
if (-not (Test-Path $ReportsDir)) { return }

# Define “base” artifacts you want to keep fewer of:
$baseLike = @(
  "wk*_*.parquet","wk*_*.csv","portfolio_v2.parquet","weights_v2.parquet","trades_v2.parquet",
  "wk6_portfolio_compare.parquet","wk6_weights_*.parquet"
)
# Define “tearsheets/images/logs” you might keep a bit more:
$tearLike = @("*.png","tearsheet*.png","compare_runs.parquet","compare_runs.csv","run_manifest.jsonl")

function Prune([string[]]$Patterns, [int]$Keep) {
  foreach ($pat in $Patterns) {
    $files = Get-ChildItem $ReportsDir -File -Filter $pat -ErrorAction SilentlyContinue |
             Sort-Object LastWriteTime -Descending
    if ($files.Count -gt $Keep) {
      $toDelete = $files | Select-Object -Skip $Keep
      $toDelete
      # uncomment to actually delete:
      # $toDelete | Remove-Item -Force
    }
  }
}

Write-Host "DRY-RUN: Base artifacts to prune (keeping $KeepBases each pattern):"
Prune -Patterns $baseLike -Keep $KeepBases | ForEach-Object { $_.FullName }

Write-Host "`nDRY-RUN: Tearsheets/logs to prune (keeping $KeepTears each pattern):"
Prune -Patterns $tearLike -Keep $KeepTears | ForEach-Object { $_.FullName }

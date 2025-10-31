[CmdletBinding()]
param([string] $RepoRoot = (Resolve-Path ".").Path)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$RepoRoot = (Resolve-Path $RepoRoot).Path
$reports = Join-Path $RepoRoot 'reports'
if (-not (Test-Path $reports)) { return }

$map = @{
  'wk0'  = @('benchmarks','gates')
  'wk3'  = @('turnover_profile','liquidity_screens')
  'wk4'  = @('voltarget_stops')
  'wk5'  = @('walkforward_dsr','canary_log')
  'wk6'  = @('portfolio_compare','factor_exposure_weekly','capacity_curve')
  'wk7'  = @('macro_gate_effect')
  'wk8'  = @('micro_effect')
  'wk11' = @('alpha_blend','ic_timeseries')
  'wk12' = @('kelly_dd')
}

$files = Get-ChildItem -LiteralPath $reports -Recurse -Force -File -ErrorAction SilentlyContinue
foreach ($f in $files) {
  $name = $f.BaseName; $ext = $f.Extension
  if ($name -match '^wk\d+_') { continue }
  foreach ($kv in $map.GetEnumerator()) {
    $wk  = $kv.Key
    foreach ($slug in $kv.Value) {
      if ($name -match $slug) {
        $newName = "$wk`_$slug$ext"
        $target = Join-Path $f.DirectoryName $newName
        if ($target -ne $f.FullName) { try { Rename-Item -LiteralPath $f.FullName -NewName $newName -Force } catch {} }
        break
      }
    }
  }
}
Write-Host "[OK] Report names standardized (where matched)." -ForegroundColor Green

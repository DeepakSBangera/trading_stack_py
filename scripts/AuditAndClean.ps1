[CmdletBinding(SupportsShouldProcess)]
param(
  [string]$RepoRoot = (Get-Location).Path,
  [string]$UniverseDir = "data_synth\prices",
  [string]$Fundamentals = "data_synth\fundamentals\fundamentals.parquet",
  [string]$ReportsDir = "reports",
  [string[]]$KeeperPatterns = @(
    '^portfolioV2PLUS_.*_L126_K4_C10',   # adjust to your winners
    '^portfolio_.*_equity'               # legacy v1 baselines
  ),
  [switch]$Plus,                         # use the enhanced v2+ backtester
  [switch]$DoPurge,                      # actually delete non-keepers in reports\
  [switch]$RefreshBaseline               # run a fresh baseline after cleanup
)

function Write-Header($text) {
  Write-Host "`n=== $text ===" -ForegroundColor Cyan
}

# 1) Sanity: key paths
Write-Header "Sanity checks"
$venv = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$tools = Join-Path $RepoRoot "tools"
$reportPs = Join-Path $tools "Report-PortfolioV2.ps1"
$pyGen = Join-Path $tools "generate_synth_data.py"
$pyV2 = Join-Path $tools "report_portfolio_v2.py"
$pyV2Plus = Join-Path $tools "report_portfolio_v2_plus.py"

$checks = @(
  @{Name="RepoRoot"; Path=$RepoRoot},
  @{Name="VenvPython"; Path=$venv},
  @{Name="Report-PortfolioV2.ps1"; Path=$reportPs},
  @{Name="generate_synth_data.py"; Path=$pyGen},
  @{Name="report_portfolio_v2.py"; Path=$pyV2},
  @{Name="report_portfolio_v2_plus.py"; Path=$pyV2Plus},
  @{Name="UniverseDir"; Path=(Join-Path $RepoRoot $UniverseDir)},
  @{Name="Fundamentals"; Path=(Join-Path $RepoRoot $Fundamentals)},
  @{Name="ReportsDir"; Path=(Join-Path $RepoRoot $ReportsDir)}
)

foreach ($item in $checks) {
  $exists = Test-Path $item.Path
  $mark = if ($exists) { "OK " } else { "MISS" }
  "{0,-28} {1}  {2}" -f $item.Name, $mark, $item.Path
}

# 2) Inventory: key folders & file counts
Write-Header "Inventory"
$treeTargets = @(".\tools", ".\configs", ".\data_synth\prices", ".\data_synth\fundamentals", ".\reports")
foreach ($t in $treeTargets) {
  if (Test-Path $t) {
    $count = (Get-ChildItem -Recurse -File $t | Measure-Object).Count
    "{0,-32} files: {1}" -f $t, $count
  } else {
    "{0,-32} (missing)" -f $t
  }
}

# 3) Light Parquet validation (names & first 2 rows)
Write-Header "Parquet quick check"
if (Test-Path $venv) {
  $pyCode = @"
import os, glob, pandas as pd
root = r'$RepoRoot'
prices_dir = os.path.join(root, r'$UniverseDir')
funds_path = os.path.join(root, r'$Fundamentals')

def head(df, n=2):
    return df.head(n).to_string(index=False)

if os.path.isdir(prices_dir):
    files = sorted(glob.glob(os.path.join(prices_dir, '*.parquet')))
    print(f'prices parquet files: {len(files)} (showing up to 3)')
    for f in files[:3]:
        try:
            df = pd.read_parquet(f)
            cols = ','.join(df.columns.tolist())
            print(' -', os.path.basename(f), 'cols=[', cols, ']')
            print(head(df))
        except Exception as e:
            print(' -', os.path.basename(f), 'ERROR:', e)
else:
    print('prices dir missing:', prices_dir)

if os.path.isfile(funds_path):
    try:
        df = pd.read_parquet(funds_path)
        cols = ','.join(df.columns.tolist())
        print('fundamentals OK cols=[', cols, ']')
        print(head(df))
    except Exception as e:
        print('fundamentals ERROR:', e)
else:
    print('fundamentals missing:', funds_path)
"@

  # Write to a temp .py and execute (compatible with PS 5.1 and 7)
  $tmpPy = Join-Path ([System.IO.Path]::GetTempPath()) ("audit_parquet_" + [System.Guid]::NewGuid().ToString() + ".py")
  Set-Content -Path $tmpPy -Value $pyCode -Encoding UTF8
  & $venv $tmpPy
  Remove-Item $tmpPy -Force -ErrorAction SilentlyContinue
} else {
  Write-Warning "Python venv not found; skipping Parquet schema print."
}

# 4) Reports: archive keepers, optionally purge others
Write-Header "Reports housekeeping"
$reportsAbs = Join-Path $RepoRoot $ReportsDir
if (-not (Test-Path $reportsAbs)) {
  Write-Host "No reports directory found; skipping archive/purge."
} else {
  $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
  $archive = Join-Path $RepoRoot "reports_archive\$stamp"
  New-Item -ItemType Directory -Force -Path $archive | Out-Null

  $all = Get-ChildItem -File (Join-Path $reportsAbs '*')
  $keepers = @()
  foreach ($f in $all) {
    $n = $f.Name
    foreach ($pat in $KeeperPatterns) {
      if ($n -match $pat) { $keepers += $f; break }
    }
  }
  "Found {0} report files; keepers matched: {1}" -f $all.Count, $keepers.Count

  if ($keepers.Count -gt 0) {
    $keepers | Copy-Item -Destination $archive -Force
    "✓ Archived keepers to $archive"
  }

  if ($DoPurge) {
    $toDelete = @()
    foreach ($f in $all) {
      if (-not ($keepers -contains $f)) { $toDelete += $f }
    }
    "Deleting {0} non-keepers..." -f $toDelete.Count
    $toDelete | Remove-Item -Force -ErrorAction SilentlyContinue
    "✓ Purged non-keepers."
  } else {
    "Dry-run: set -DoPurge to actually delete non-keepers."
  }
}

# 5) Optional: re-run a clean baseline
if ($RefreshBaseline) {
  Write-Header "Baseline backtest"
  $ps = Join-Path $tools "Report-PortfolioV2.ps1"
  if (-not (Test-Path $ps)) {
    Write-Error "Missing $ps; create it first."
  } else {
    $params = @{
      UniverseDir = $UniverseDir
      Start       = "2015-01-01"
      Lookback    = 126
      TopN        = 4
      Rebalance   = "ME"
      CostBps     = 10
      OutDir      = $ReportsDir
    }
    if ($Plus) { $params["Plus"] = $true }
    & $ps @params
    if ($LASTEXITCODE -ne 0) { throw "Baseline run failed with code $LASTEXITCODE" }
  }
}

# 6) Summary
Write-Header "Summary"
"Universe dir: $UniverseDir"
"Fundamentals: $Fundamentals"
"Reports dir : $ReportsDir"
"Keepers regex:"
$KeeperPatterns | ForEach-Object { " - $_" }
"Done."

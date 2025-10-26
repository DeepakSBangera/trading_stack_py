# === scripts\Cleanup-Repo.ps1 (v2: protected roots + preview limit) ============
[CmdletBinding()]
param(
  [string]$Root = ".",
  [string]$Reports = "reports",
  [switch]$ArchiveBeforeDelete,
  [int]$KeepRecentBases = 3,
  [switch]$IncludeGovernance,
  [switch]$DeleteCsv,
  [switch]$DryRun,
  [int]$PreviewLimit = 50   # max items to preview per section
)

$ErrorActionPreference = "Stop"
$here = (Resolve-Path $Root).Path
$ReportsDir = Join-Path $here $Reports

function Info($m,[ConsoleColor]$c=[ConsoleColor]::Cyan){$o=$Host.UI.RawUI.ForegroundColor;$Host.UI.RawUI.ForegroundColor=$c;Write-Host $m;$Host.UI.RawUI.ForegroundColor=$o}
function Warn($m){Info $m ([ConsoleColor]::Yellow)}
function Good($m){Info $m ([ConsoleColor]::Green)}
function Bad ($m){Info $m ([ConsoleColor]::Red)}

# ---------- PROTECTED ROOTS (never touch) --------------------------------------
$ProtectedRoots = @(
  (Join-Path $here ".venv"),
  (Join-Path $here ".venv-test"),
  (Join-Path $here "venv"),
  (Join-Path $here "env"),
  (Join-Path $here "node_modules")
) | ForEach-Object { $_.ToLowerInvariant() }

function Is-Protected([string]$path) {
  $p = $path.ToLowerInvariant()
  foreach ($root in $ProtectedRoots) {
    if ($p -like ($root + "*")) { return $true }
  }
  return $false
}

if (-not (Test-Path $ReportsDir)) { New-Item -ItemType Directory -Force -Path $ReportsDir | Out-Null }

function Get-BaseName([string]$name){
  $n = [System.IO.Path]::GetFileNameWithoutExtension($name)
  foreach($s in "_weights","_trades","_tearsheet"){ if ($n.EndsWith($s)) { $n = $n.Substring(0,$n.Length-$s.Length) } }
  return $n
}

# ---------- Phase out CSVs (convert then delete) --------------------------------
$py = ".\.venv\Scripts\python.exe"
$converter = ".\tools\report_to_parquet.py"
$csvs = Get-ChildItem -File (Join-Path $ReportsDir "*.csv") -ErrorAction SilentlyContinue
if ($csvs) {
  if (Test-Path $py -and Test-Path $converter) {
    Info "Converting CSV → Parquet (then deleting CSVs)..." ([ConsoleColor]::Magenta)
    $argsList = @($converter,"--reports",$ReportsDir,"--include-governance","--delete-csv")
    if ($DryRun){ Warn "DryRun: would run: $py $($argsList -join ' ')" } else { & $py @argsList }
  } elseif ($DeleteCsv) {
    Warn "CSV converter not found; -DeleteCsv supplied → will delete CSVs directly."
    foreach($f in $csvs){ if ($DryRun){ Write-Host "[DryRun] DEL $($f.FullName)" } else { Remove-Item $f.FullName -Force -ErrorAction SilentlyContinue } }
  } else {
    Warn "CSV files present but converter not found. Re-run later or use -DeleteCsv to remove directly."
  }
}

# Refresh files
$reportFiles = Get-ChildItem -File $ReportsDir -Recurse -ErrorAction SilentlyContinue

# ---------- Choose portfolio bases to keep -------------------------------------
$portfolio = $reportFiles | Where-Object { $_.Name -like "portfolioV2_*" -and $_.Extension -in ".parquet",".png" }
$groups = @{}
foreach($f in $portfolio){
  $b = Get-BaseName $f.Name
  if (-not $groups.ContainsKey($b)) { $groups[$b] = @() }
  $groups[$b] += $f
}
$orderedBases = $groups.Keys | Sort-Object {
  ($groups[$_]|Sort-Object LastWriteTime -Descending|Select-Object -First 1).LastWriteTime
} -Descending

$keepBases = $orderedBases | Select-Object -First $KeepRecentBases
$keepSet = New-Object System.Collections.Generic.HashSet[string]

# keep all tearsheets
$tears = $portfolio | Where-Object { $_.BaseName -like "*_tearsheet" }
foreach($f in $tears){ [void]$keepSet.Add($f.FullName) }
# keep artifacts for chosen bases
foreach($b in $keepBases){ foreach($f in $groups[$b]){ [void]$keepSet.Add($f.FullName) } }

# candidates for deletion (reports only)
$candidateReportDeletes = @()
foreach ($f in $portfolio) {
  if (-not $keepSet.Contains($f.FullName)) { $candidateReportDeletes += $f }
}

# ---------- Governance pruning (wk*) -------------------------------------------
$gov = $reportFiles | Where-Object { $_.DirectoryName -eq $ReportsDir -and ($_.Name -like "wk*.parquet" -or $_.Name -like "wk*.png" -or $_.Name -like "wk*.csv") }
$govDeletes = @()
if ($IncludeGovernance) {
  $byStem = @{}
  foreach($g in $gov){
    $stem = [System.IO.Path]::GetFileNameWithoutExtension($g.Name)
    if (-not $byStem.ContainsKey($stem)) { $byStem[$stem] = @() }
    $byStem[$stem] += $g
  }
  foreach($k in $byStem.Keys){
    $keepers = $byStem[$k] | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    $trash   = $byStem[$k] | Where-Object { $_.FullName -ne $keepers[0].FullName }
    $govDeletes += $trash
  }
}

# ---------- Junk caches/files, but NEVER inside protected roots -----------------
$junkDirs = @("__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".ipynb_checkpoints", ".benchmarks")
$junkFiles = @("*.pyc","*.pyo","*.tmp","*.bak","*.orig","Thumbs.db","desktop.ini",".DS_Store")

Info "Scanning for junk caches/files under $here (protected roots skipped)..." ([ConsoleColor]::Gray)
$allDirs  = Get-ChildItem -Recurse -Directory -ErrorAction SilentlyContinue $here
$dirHits  = $allDirs | Where-Object { ($junkDirs -contains $_.Name) -and -not (Is-Protected $_.FullName) }

$fileHits = @()
foreach($pat in $junkFiles){
  $fileHits += Get-ChildItem -Recurse -File -ErrorAction SilentlyContinue -Include $pat -Path $here | Where-Object { -not (Is-Protected $_.FullName) }
}
$fileHits = $fileHits | Sort-Object FullName -Unique

# ---------- Preview helpers -----------------------------------------------------
function Preview($label, $items) {
  if (-not $items -or $items.Count -eq 0) { return }
  Write-Host ""
  Warn "$label  (showing up to $PreviewLimit of $($items.Count))"
  $i = 0
  foreach($x in $items){
    if ($i -ge $PreviewLimit) { Write-Host "... (truncated)"; break }
    Write-Host ("[DryRun] {0}" -f ($x.FullName))
    $i++
  }
}

# ---------- Archive (optional) --------------------------------------------------
$archiveDir = $null
$toDelete = @()
$toDelete += $candidateReportDeletes
$toDelete += $govDeletes
if ($ArchiveBeforeDelete -and $toDelete.Count -gt 0) {
  $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
  $archiveDir = Join-Path $ReportsDir ("reports_archive\{0}" -f $stamp)
  if (-not $DryRun) { New-Item -ItemType Directory -Force -Path $archiveDir | Out-Null }
  Info "Archiving before delete → $archiveDir"
  foreach($f in ($toDelete | Sort-Object FullName -Unique)){
    $dest = Join-Path $archiveDir $f.Name
    if ($DryRun){ Write-Host "[DryRun] COPY $($f.FullName) -> $dest" } else { Copy-Item $f.FullName $dest -Force -ErrorAction SilentlyContinue }
  }
}

# ---------- Delete --------------------------------------------------------------
if ($DryRun){
  Preview "Would delete report artifacts:" ($toDelete | Sort-Object FullName -Unique)
  Preview "Would remove cache directories:" $dirHits
  Preview "Would remove stray files:" $fileHits
  Warn "`nDryRun was ON. Re-run without -DryRun to apply."
} else {
  if ($toDelete){ Info "Deleting report artifacts..." ([ConsoleColor]::Red); ($toDelete|Sort-Object FullName -Unique) | ForEach-Object { Remove-Item $_.FullName -Force -ErrorAction SilentlyContinue } }
  if ($dirHits){ Info "Removing cache directories..." ([ConsoleColor]::Red); foreach($d in $dirHits){ Remove-Item $d.FullName -Recurse -Force -ErrorAction SilentlyContinue } }
  if ($fileHits){ Info "Removing stray files..." ([ConsoleColor]::Red); foreach($f in $fileHits){ Remove-Item $f.FullName -Force -ErrorAction SilentlyContinue } }
  Good "Cleanup applied."
}

# ---------- Summary -------------------------------------------------------------
Write-Host ""
Good "Summary:"
"{0,-28} {1,8}" -f "Kept bases:", $keepBases.Count | Write-Host
"{0,-28} {1,8}" -f "Deleted report files:", ($toDelete.Count) | Write-Host
"{0,-28} {1,8}" -f "Cache dirs removed:", ($dirHits.Count) | Write-Host
"{0,-28} {1,8}" -f "Stray files removed:", ($fileHits.Count) | Write-Host
if ($archiveDir){ "{0,-28} {1}" -f "Archived to:", $archiveDir | Write-Host }

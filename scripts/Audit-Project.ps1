[CmdletBinding()]
param(
  [string]$RepoRoot = (Get-Location).Path
)

function New-Stamp { (Get-Date -Format "yyyyMMdd_HHmmss") }
function Ensure-Dir([string]$p) { if (-not (Test-Path $p)) { New-Item -ItemType Directory -Force -Path $p | Out-Null }; return $p }
function BoolStr([bool]$b) { if ($b) { return "OK" } else { return "MISS" } }
function Exists([string]$root, [string]$rel) { Test-Path (Join-Path $root $rel) }
function Tern([bool]$Cond, $IfVal, $ElseVal) { if ($Cond) { return $IfVal } else { return $ElseVal } }

# ---------- Paths ----------
$Root = (Resolve-Path $RepoRoot).Path
$Stamp = New-Stamp
$AuditRoot = Ensure-Dir (Join-Path $Root ("reports\audit\" + $Stamp))

$VenvPy = Join-Path $Root ".venv\Scripts\python.exe"
$Tools  = Join-Path $Root "tools"
$Reports= Ensure-Dir (Join-Path $Root "reports")

# config / configs (use both if present)
$ConfigDirs = @()
$cfg1 = Join-Path $Root "config"
$cfg2 = Join-Path $Root "configs"
if (Test-Path $cfg1) { $ConfigDirs += $cfg1 }
if (Test-Path $cfg2) { $ConfigDirs += $cfg2 }

Write-Host "=== Audit: $Root ==="

# ---------- 1) Full file inventory ----------
Write-Host "`n[1/6] Scanning files & directories ..."
$files = Get-ChildItem -Recurse -File -Force $Root -ErrorAction SilentlyContinue |
  Select-Object `
    @{n="path";e={$_.FullName}},
    @{n="relpath";e={$_.FullName.Substring($Root.Length).TrimStart('\')}},
    @{n="name";e={$_.Name}},
    @{n="ext";e={$_.Extension.ToLower()}},
    @{n="bytes";e={$_.Length}},
    @{n="last_write_utc";e={$_.LastWriteTimeUtc.ToString("o")}}

$dirsTop = Get-ChildItem -Directory $Root -ErrorAction SilentlyContinue |
  Select-Object FullName, Name

$filesCsv = Join-Path $AuditRoot "files_inventory.csv"
$dirsCsv  = Join-Path $AuditRoot "dirs_top.csv"
$files | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $filesCsv
$dirsTop | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $dirsCsv
Write-Host ("  Files: {0}  → {1}" -f $files.Count, $filesCsv)
Write-Host ("  Top-level dirs: {0}  → {1}" -f $dirsTop.Count, $dirsCsv)

# ---------- 2) Summaries ----------
Write-Host "`n[2/6] Building summaries ..."

# by extension (no inline-if)
$extSummary = @()
$groups = $files | Group-Object ext
foreach ($g in $groups) {
  $sumBytes = ($g.Group | Measure-Object -Property bytes -Sum).Sum
  $extName = (Tern ($null -ne $g.Name -and $g.Name -ne "") $g.Name "(none)")
  $extSummary += [pscustomobject]@{ ext = $extName; files = $g.Count; bytes = $sumBytes }
}
$extSummary = $extSummary | Sort-Object -Property bytes -Descending
$extCsv = Join-Path $AuditRoot "summary_by_extension.csv"
$extSummary | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $extCsv
Write-Host ("  Summary by extension → {0}" -f $extCsv)

# by top folder
$topSummary = @()
foreach ($d in $dirsTop) {
  $prefix = $d.FullName
  $under  = $files | Where-Object { $_.path -like "$prefix*" }
  $sumB   = ($under | Measure-Object -Property bytes -Sum).Sum
  $topSummary += [pscustomobject]@{ folder = $d.Name; files = $under.Count; bytes = $sumB }
}
$tsCsv = Join-Path $AuditRoot "summary_by_top_folder.csv"
$topSummary | Sort-Object -Property bytes -Descending | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $tsCsv
Write-Host ("  Summary by top folder → {0}" -f $tsCsv)

# ---------- 3) Reports overview ----------
Write-Host "`n[3/6] Reports overview ..."
$reportFiles = Get-ChildItem -File $Reports -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending
$repCsv = Join-Path $AuditRoot "reports_latest.csv"
$reportFiles | Select-Object `
  @{n="name";e={$_.Name}},
  @{n="bytes";e={$_.Length}},
  @{n="last_write_utc";e={$_.LastWriteTimeUtc.ToString("o")}} |
  Export-Csv -NoTypeInformation -Encoding UTF8 -Path $repCsv
Write-Host ("  Reports found: {0}  → {1}" -f $reportFiles.Count, $repCsv)

# ---------- 4) Data overview (Parquet counts/bytes only) ----------
Write-Host "`n[4/6] Data overview (Parquet counts/bytes only) ..."
$dataRoots = @()
$dataRoots += (Join-Path $Root "data_synth")
$dataRoots += (Join-Path $Root "data")
# uniq & existing
$tmpRoots = @()
foreach ($dr in $dataRoots) { if (Test-Path $dr) { $tmpRoots += $dr } }
$dataRoots = $tmpRoots | Select-Object -Unique

$dataRows = @()
foreach ($dr in $dataRoots) {
  $parqs = Get-ChildItem -Recurse -File $dr -Filter *.parquet -ErrorAction SilentlyContinue
  $bytes = 0
  foreach ($p in $parqs) { $bytes += $p.Length }
  $dataRows += [pscustomobject]@{ root=$dr; files=$parqs.Count; bytes=$bytes }
}
$dataCsv = Join-Path $AuditRoot "data_overview.csv"
$dataRows | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $dataCsv
Write-Host ("  Data roots scanned: {0}  → {1}" -f $dataRoots.Count, $dataCsv)

# ---------- 5) Milestones (presence-only) ----------
Write-Host "`n[5/6] Milestones (presence-only) ..."

# helper checks
$SynthPricesOk = $false
$sp = Join-Path $Root "data_synth\prices"
if (Test-Path $sp) {
  $cnt = (Get-ChildItem -File $sp -Filter *.parquet -ErrorAction SilentlyContinue | Measure-Object).Count
  if ($cnt -gt 0) { $SynthPricesOk = $true }
}
$SynthFundsOk = $false
$sf = Join-Path $Root "data_synth\fundamentals"
if (Test-Path $sf) {
  $cntf = (Get-ChildItem -File $sf -Filter *.parquet -ErrorAction SilentlyContinue | Measure-Object).Count
  if ($cntf -gt 0) { $SynthFundsOk = $true }
}
$TearCsvOk = $false
$cntT = (Get-ChildItem -File $Reports -Filter "*_tearsheet.csv" -ErrorAction SilentlyContinue | Measure-Object).Count
if ($cntT -gt 0) { $TearCsvOk = $true }

$milestones = @(
  @{ name="venv python";                         present=(Test-Path $VenvPy) },
  @{ name="tools/report_portfolio_v2.py";        present=(Exists $Root "tools\report_portfolio_v2.py") },
  @{ name="tools/Report-PortfolioV2.ps1";        present=(Exists $Root "tools\Report-PortfolioV2.ps1") },
  @{ name="tools/Report-Portfolio.ps1 (v1)";     present=(Exists $Root "tools\Report-Portfolio.ps1") },
  @{ name="Synthetic prices present";            present=$SynthPricesOk },
  @{ name="Synthetic fundamentals present";      present=$SynthFundsOk },
  @{ name="tools/make_tearsheet.py";             present=(Exists $Root "tools\make_tearsheet.py") },
  @{ name="Any *_tearsheet.csv in reports";      present=$TearCsvOk },
  @{ name="tools/build_catalog.py";              present=(Exists $Root "tools\build_catalog.py") },
  @{ name="tools/pit_join.py";                   present=(Exists $Root "tools\pit_join.py") },
  @{ name="tools/emit_manifest.py";              present=(Exists $Root "tools\emit_manifest.py") },
  @{ name="scripts/Run-Baseline.ps1";            present=(Exists $Root "scripts\Run-Baseline.ps1") },
  @{ name="config or configs dir";               present=($ConfigDirs.Count -gt 0) }
)

$msCsv = Join-Path $AuditRoot "milestones.csv"
# print to console
foreach ($m in $milestones) {
  $status = BoolStr $m.present
  "{0,-36} {1}" -f $m.name, $status | Write-Host
}
# write csv
$milestones | ForEach-Object {
  [pscustomobject]@{ milestone=$_.name; status=(BoolStr $_.present) }
} | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $msCsv
Write-Host ("  Milestones → {0}" -f $msCsv)

# ---------- 6) Hours accounting (presence → fraction) ----------
Write-Host "`n[6/6] Hours accounting (presence-based) ..."

# Planned hours (midpoints)
$HoursPlan = @(
  @{ key="Pilot_DataLayer";     hours=5 },
  @{ key="Pilot_Factors";       hours=7 },
  @{ key="Pilot_Portfolio";     hours=5 },
  @{ key="Pilot_EvalPack";      hours=7 },
  @{ key="Pilot_TestsDocs";     hours=4 },
  @{ key="MVP_LiveAdapters";    hours=9 },
  @{ key="MVP_UniverseBuilder"; hours=7 },
  @{ key="MVP_Orchestration";   hours=9 },
  @{ key="MVP_Robustness";      hours=10 },
  @{ key="Prod_KiteIntegr";     hours=10 },
  @{ key="Prod_OrderPath";      hours=15 },
  @{ key="Prod_Monitoring";     hours=15 },
  @{ key="Prod_CostLatency";    hours=10 },
  @{ key="Modeling_Validation"; hours=50 }
)

# Fractions (strictly from presence) — compute with plain if/else
$credit = @{}

if ($SynthPricesOk) { $credit["Pilot_DataLayer"] = 1.0 } else { $credit["Pilot_DataLayer"] = 0.0 }
if (Exists $Root "tools\report_portfolio_v2.py") { $credit["Pilot_Factors"] = 1.0 } else { $credit["Pilot_Factors"] = 0.0 }
if (Exists $Root "tools\Report-PortfolioV2.ps1") { $credit["Pilot_Portfolio"] = 1.0 } else { $credit["Pilot_Portfolio"] = 0.0 }
if ((Exists $Root "tools\make_tearsheet.py") -and $TearCsvOk) { $credit["Pilot_EvalPack"] = 1.0 } else { $credit["Pilot_EvalPack"] = 0.0 }
if (Test-Path (Join-Path $Root "tests")) { $credit["Pilot_TestsDocs"] = 0.5 } else { $credit["Pilot_TestsDocs"] = 0.0 }
$credit["MVP_LiveAdapters"] = 0.0
if ($ConfigDirs.Count -gt 0) { $credit["MVP_UniverseBuilder"] = 0.5 } else { $credit["MVP_UniverseBuilder"] = 0.0 }
if (Exists $Root "scripts\Run-Baseline.ps1") { $credit["MVP_Orchestration"] = 0.5 } else { $credit["MVP_Orchestration"] = 0.0 }
$credit["MVP_Robustness"] = 0.0
$credit["Prod_KiteIntegr"] = 0.0
$credit["Prod_OrderPath"] = 0.0
if (Exists $Root "tools\emit_manifest.py") { $credit["Prod_Monitoring"] = 0.25 } else { $credit["Prod_Monitoring"] = 0.0 }
$credit["Prod_CostLatency"] = 0.0
$PortfolioCsvCount = (Get-ChildItem -File $Reports -Filter "portfolioV2_*.csv" -ErrorAction SilentlyContinue | Measure-Object).Count
if ($PortfolioCsvCount -gt 0) { $credit["Modeling_Validation"] = 0.1 } else { $credit["Modeling_Validation"] = 0.0 }

# Totals
$total = 0.0; $done = 0.0
$hoursRows = @()
foreach ($h in $HoursPlan) {
  $k = $h.key; $hrs = [double]$h.hours
  $frac = 0.0
  if ($credit.ContainsKey($k)) { $frac = [double]$credit[$k] }
  $total += $hrs
  $done  += ($hrs * $frac)
  $hoursRows += [pscustomobject]@{
    task = $k; planned_hours = $hrs; credited_fraction = $frac; credited_hours = ($hrs * $frac)
  }
}
$remain = [math]::Max(0.0, $total - $done)
Write-Host ("  Total planned:     {0,6:N1} h" -f $total)
Write-Host ("  Completed (cred.): {0,6:N1} h" -f $done)
Write-Host ("  Remaining (est.):  {0,6:N1} h" -f $remain)

$hoursCsv  = Join-Path $AuditRoot "hours_accounting.csv"
$hoursJson = Join-Path $AuditRoot "hours_accounting.json"
$hoursRows | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $hoursCsv
([pscustomobject]@{ total=$total; completed=$done; remaining=$remain; rows=$hoursRows }) |
  ConvertTo-Json -Depth 5 | Set-Content -Path $hoursJson -Encoding UTF8

# ---------- Index file ----------
$indexTxt = Join-Path $AuditRoot "_INDEX.txt"
$idx = @()
$idx += "Audit timestamp: $Stamp"
$idx += "Repo root:       $Root"
$idx += ""
$idx += "Artifacts:"
$idx += " - Files inventory:              $((Resolve-Path $filesCsv).Path)"
$idx += " - Top-level dirs:               $((Resolve-Path $dirsCsv).Path)"
$idx += " - Summary by extension:         $((Resolve-Path $extCsv).Path)"
$idx += " - Summary by top folder:        $((Resolve-Path $tsCsv).Path)"
$idx += " - Reports listing:              $((Resolve-Path $repCsv).Path)"
$idx += " - Data overview (roots):        $((Resolve-Path $dataCsv).Path)"
$idx += " - Milestones:                   $((Resolve-Path $msCsv).Path)"
$idx += " - Hours accounting (CSV):       $((Resolve-Path $hoursCsv).Path)"
$idx += " - Hours accounting (JSON):      $((Resolve-Path $hoursJson).Path)"
$idx | Set-Content -Path $indexTxt -Encoding UTF8

Write-Host "`n✓ Audit complete. See: $AuditRoot"

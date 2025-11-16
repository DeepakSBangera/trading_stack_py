# ===================== tools\Audit-W0W44.ps1 =====================
[CmdletBinding()]
param(
  [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
  [string]$OutDir   = "$PSScriptRoot\..\reports\audit",
  [switch]$WriteAnchor
)

$ErrorActionPreference = "Stop"

function Ensure-Dir([string]$p){
  (Resolve-Path -LiteralPath (New-Item -ItemType Directory -Force -Path $p)).Path
}

# --- entrypoint detection (safe regex) ---
$MainGuardRegex = @'
if\s+__name__\s*==\s*["\']__main__["\']
'@
function Test-PyEntrypoint {
  param([Parameter(Mandatory)][string]$Path)
  Select-String -LiteralPath $Path -Pattern $MainGuardRegex -Quiet -ErrorAction SilentlyContinue
}

# --- import sniff (first 200 lines) ---
$ImportRegex1 = '^\s*import\s+([A-Za-z0-9_\.]+)'
$ImportRegex2 = '^\s*from\s+([A-Za-z0-9_\.]+)\s+import\s+'
function Get-TopImports {
  param([Parameter(Mandatory)][string]$Path)
  try {
    $text = Get-Content -LiteralPath $Path -TotalCount 200 -ErrorAction Stop
  } catch { return @() }
  $mods = @()
  foreach($line in $text){
    $m1 = [regex]::Match($line, $ImportRegex1)
    if($m1.Success){
      $mods += $m1.Groups[1].Value.Split('.')[0]
      continue
    }
    $m2 = [regex]::Match($line, $ImportRegex2)
    if($m2.Success){
      $mods += $m2.Groups[1].Value.Split('.')[0]
      continue
    }
  }
  $mods | Where-Object { $_ -and $_.Trim() -ne "" } | Sort-Object -Unique
}

# --- local package names present in repo ---
$LocalRoots = @("modules","app","src","scripts")
$LocalPkgs  = @()
foreach($r in $LocalRoots){
  $p = Join-Path $RepoRoot $r
  if(Test-Path $p){
    $LocalPkgs += (Get-ChildItem -Path $p -Directory -ErrorAction SilentlyContinue |
                   Select-Object -ExpandProperty Name)
  }
}
$LocalPkgs = $LocalPkgs | Sort-Object -Unique

function Test-LocalDependency {
  param([string[]]$Imports)
  $hits = @()
  foreach($i in $Imports){
    if($LocalPkgs -contains $i){ $hits += $i }
  }
  $hits | Sort-Object -Unique
}

# --- expected artifacts per week ---
$WeekGlobs = [ordered]@{
  "W0"  = @("reports\wk0_gates.csv","reports\benchmarks.csv","docs\process_notes.md")
  "W1"  = @("reports\wk1_entry_exit_baseline.csv")
  "W2"  = @("reports\wk2_qc_summary.csv")
  "W2A" = @("docs\w2a_kite_data_toggle.md")
  "W2B" = @("docs\w2b_mobile_monitor.md")
  "W2C" = @("reports\wk2c_pit_sanity.csv")
  "W3"  = @("reports\wk3_turnover_profile.csv","reports\liquidity_screens.csv","docs\capacity_policy.md")
  "W4"  = @("reports\wk4_voltarget_stops.csv","config\kill_switch.yaml","docs\kill_switch_matrix.md")
  "W5"  = @("reports\wk5_walkforward_dsr.csv","reports\canary_log.csv")
  "W6"  = @("reports\wk6_portfolio_compare.csv","reports\factor_exposure_weekly.csv","reports\capacity_curve.csv")
  "W7"  = @("reports\wk7_macro_gate_effect.csv")
  "W8"  = @("reports\wk8_micro_effect.csv")
  "W9"  = @("reports\wk9_pricing_causal.csv")
  "W10" = @("reports\wk10_forecast_eval.csv")
  "W11" = @("reports\wk11_alpha_blend.csv","reports\ic_timeseries.csv")
  "W12" = @("reports\wk12_kelly_dd.csv")
  "W13" = @("reports\wk13_pairs_oos.csv")
  "W14" = @("reports\wk14_vol_models.csv")
  "W15" = @("reports\wk15_regime_policy.csv","docs\aggressive_year_policy.md","reports\aggressive_switch_log.csv")
  "W16" = @("reports\wk16_exec_calibration.csv")
  "W17" = @("reports\signal_half_life.csv","reports\kill_switch_tests.csv","reports\wk17_ops_attrib.csv")
  "W18" = @("reports\wk18_pricing_tilts.csv")
  "W19" = @("reports\wk19_short_spreads.csv")
  "W20" = @("reports\wk20_after_tax_view.csv")
  "W21" = @("reports\wk21_purgedcv_pbo.csv")
  "W22" = @("reports\wk22_pit_integrity.csv")
  "W23" = @("reports\wk23_factor_risk_model.csv","docs\sector_factor_caps.md")
  "W24" = @("reports\wk24_black_litterman_compare.csv")
  "W25" = @("reports\wk25_exec_engineering.csv")
  "W26" = @("reports\wk26_ops_cvar.csv")
  "W27" = @("reports\wk27_bandit_selection.csv")
  "W28" = @("reports\wk28_ope_results.csv")
  "W29" = @("reports\wk29_safe_policy.csv")
  "W30" = @("reports\wk30_risk_sens_policy.csv")
  "W31" = @("reports\wk31_exec_bandit.csv")
  "W32" = @("reports\wk32_offline_rl_sizing.csv")
  "W33" = @("reports\wk33_barbell_results.csv")
  "W34" = @("reports\wk34_leverage_throttle.csv")
  "W35" = @("reports\wk35_tca_tuning.csv")
  "W36" = @("reports\wk36_options_overlay.csv")
  "W37" = @("reports\wk37_alpha_table.csv")
  "W38" = @("reports\wk38_turnover_profile.csv")
  "W39" = @("reports\wk39_capacity_audit.csv")
  "W40" = @("reports\wk40_exec_quality.csv")
  "W41" = @("reports\wk41_momentum_tilt.csv")
  "W42" = @("reports\wk42_after_tax_schedule.csv")
  "W43" = @("reports\wk43_barbell_compare.csv")
  "W44" = @("docs\wk44_redteam_report.md")
  "W45" = @(
    "reports\wk45_end_to_end.csv",
    "reports\wk45_final_snapshot.zip"
  )
}
$GovernanceDocs = @(
  "docs\change_log_template.md",
  "docs\release_checklist.md",
  "docs\kill_switch_matrix.md",
  "config\kill_switch.yaml",
  "docs\capacity_policy.md",
  "docs\sector_factor_caps.md",
  "docs\ic_promotion_rules.md",
  "docs\data_lineage.md",
  "docs\pretrade_checklist.md",
  "docs\aggressive_year_policy.md"
)

# --- gather python files ---
$pyRoots = @("scripts","modules","app","src")
$pyFiles = @()
foreach($root in $pyRoots){
  $rp = Join-Path $RepoRoot $root
  if(Test-Path $rp){
    $pyFiles += Get-ChildItem -Path $rp -Recurse -Filter *.py -File -ErrorAction SilentlyContinue
  }
}
$pyFiles = $pyFiles | Sort-Object FullName -Unique

# --- analyze scripts ---
$scriptRows = New-Object System.Collections.Generic.List[object]
foreach($f in $pyFiles){
  $rel = $f.FullName.Substring($RepoRoot.Length).TrimStart('\','/')
  $imports = Get-TopImports -Path $f.FullName
  $localDeps = Test-LocalDependency -Imports $imports
  $isEntrypoint = Test-PyEntrypoint -Path $f.FullName
  $scriptRows.Add([pscustomobject]@{
    file          = $rel
    entrypoint    = [bool]$isEntrypoint
    imports       = ($imports -join ',')
    local_deps    = ($localDeps -join ',')
    size_bytes    = $f.Length
    modified_utc  = $f.LastWriteTimeUtc.ToString("s") + "Z"
  })
}

# --- governance presence ---
$govRows = New-Object System.Collections.Generic.List[object]
foreach($g in $GovernanceDocs){
  $p = Join-Path $RepoRoot $g
  $govRows.Add([pscustomobject]@{ artifact = $g; exists = Test-Path $p })
}

# --- week status (expected artifacts) ---
function Test-GlobList {
  param([string[]]$Globs)
  $hits = 0
  foreach($g in $Globs){
    if(Test-Path (Join-Path $RepoRoot $g)){ $hits++ }
  }
  if($hits -eq 0){ return "Missing" }
  if($hits -lt $Globs.Count){ return "Partial" }
  return "Done"
}
$weekRows = New-Object System.Collections.Generic.List[object]
foreach($k in $WeekGlobs.Keys){
  $weekRows.Add([pscustomobject]@{
    week     = $k
    expected = ($WeekGlobs[$k] -join '; ')
    status   = (Test-GlobList -Globs $WeekGlobs[$k])
  })
}

# --- tests ---
$pytestIni = Test-Path (Join-Path $RepoRoot "pytest.ini")
$pyproject = Test-Path (Join-Path $RepoRoot "pyproject.toml")
$testsDir  = Test-Path (Join-Path $RepoRoot "tests")

# --- outputs ---
$out = Ensure-Dir $OutDir
$auditScriptsCsv         = Join-Path $out "audit_scripts.csv"
$auditWeeksCsv           = Join-Path $out "audit_weeks.csv"
$auditSummary            = Join-Path $out "audit_summary.json"
$auditScriptsWithWeekCsv = Join-Path $out "audit_scripts_with_week.csv"
$anchorMd                = Join-Path $out "ANCHOR_SCRIPTS.md"

$scriptRows | Export-Csv -NoTypeInformation -Path $auditScriptsCsv -Encoding UTF8
$weekRows   | Export-Csv -NoTypeInformation -Path $auditWeeksCsv   -Encoding UTF8

$summaryObj = [pscustomobject]@{
  repo_root       = $RepoRoot
  generated_utc   = (Get-Date).ToUniversalTime().ToString("s") + "Z"
  total_scripts   = $scriptRows.Count
  entrypoints     = ($scriptRows | Where-Object entrypoint | Measure-Object).Count
  pytest_ini      = $pytestIni
  pyproject_toml  = $pyproject
  tests_dir       = $testsDir
  governance      = $govRows
  week_status     = $weekRows
}
$summaryObj | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $auditSummary -Encoding UTF8

# --- derived script weeks (filename-based, read-only) ---
# Reclassify weeks (catch W6, W2A, wk6_, wk4-, etc.)
$deriveWeek = {
  param($p)

  # Normalize to backslashes
  $norm = $p -replace '/', '\'

  # 1) Match "\W6_", "\W10.", "\W2A\" etc.
  #    Example: "scripts\W6_portfolio_compare.py" -> W6
  if ($norm -match '(?i)[\\](w\d{1,2}[A-Z]?)(?=[_\\\.]|$)') {
    return $matches[1].ToUpper()
  }
  # 2) Match "\wk6_", "\wk10-utils", "\wk4.test" etc.
  #    Example: "modules\wk6_utils\foo.py" -> W6
  elseif ($norm -match '(?i)[\\](wk)(\d{1,2})([A-Z]?)(?=[_\\\.-]|$)') {
    return ("W{0}{1}" -f $matches[2], $matches[3].ToUpper())
  }
  # 3) Fallback: any standalone "w6", "w10a" token at segment boundaries
  #    Example: "scripts\misc\w12_cleanup.py" -> W12
  elseif ($norm -match '(?i)(^|\\)(w\d{1,2}[A-Z]?)([_\\\.]|$)') {
    return $matches[2].ToUpper()
  }
  # 4) No detectable week pattern
  else {
    return ""
  }
}

$withWeek = $scriptRows | ForEach-Object {
  $dw = & $deriveWeek $_.file
  [pscustomobject]@{
    file       = $_.file
    entrypoint = $_.entrypoint
    week       = $dw
  }
}

$withWeek | Export-Csv -NoTypeInformation -Path $auditScriptsWithWeekCsv -Encoding UTF8

# --- KPIs for derived week mapping ---
$untag = $withWeek | Where-Object { -not $_.week }

Write-Host "Derived week mapping (from filenames) - KPIs:" -ForegroundColor Yellow
Write-Host ("Untagged (before fix you saw 105): {0}" -f @($untag).Count)
Write-Host ("Untagged entrypoints: {0}" -f @($untag | Where-Object { $_.entrypoint -eq $true }).Count)
Write-Host "Untagged by top folder:"

$untag |
  ForEach-Object { ($_."file" -split '\\')[0] } |
  Group-Object |
  Sort-Object Count -Descending |
  ForEach-Object { Write-Host ("  {0}: {1}" -f $_.Name, $_.Count) }

Write-Host "Files that LOOK weeky (contain 'wk' + digits) but still untagged:"
$untag |
  Where-Object { $_.file -match '(?i)\\wk\d' } |
  Select-Object -First 20 file |
  Format-Table -AutoSize

Write-Host "Scripts per derived week (top 10):"
$withWeek |
  Group-Object week |
  Where-Object Name |
  Sort-Object Count -Descending |
  Select-Object -First 10 |
  ForEach-Object { Write-Host ("  {0}: {1}" -f $_.Name, $_.Count) }

# --- anchor (stable string formatting) ---
if($WriteAnchor){
  $hdr = @"
# Anchor - Scripts Overview (Auto-Generated)

- Repo: $RepoRoot
- Generated (UTC): $((Get-Date).ToUniversalTime().ToString("s"))Z
- Total scripts: $($scriptRows.Count)
- Entrypoints: $((($scriptRows | Where-Object entrypoint).Count))

This file is the living index. Append more details below as we analyze each file.

## Scripts (grouped by top folder)

"@

  $body = ""
  $byFolder = $scriptRows | ForEach-Object {
    $first = (($_.file) -replace '\\','/') -split '/' | Select-Object -First 1
    [pscustomobject]@{ folder=$first; row=$_ }
  } | Group-Object folder | Sort-Object Name

  foreach($g in $byFolder){
    $body += ("### {0}`r`n`r`n" -f $g.Name)
    foreach($it in $g.Group){
      $r = $it.row
      $body += ("- {0}`r`n" -f $r.file)
      $body += ("  - Entrypoint: {0}; Imports: {1}; Local deps: {2}; Size: {3} KB; Mod(UTC): {4}`r`n" -f `
                 $r.entrypoint, $r.imports, $r.local_deps, [math]::Round($r.size_bytes/1024,1), $r.modified_utc)
      $body += ("  - What it does: (to be filled)`r`n")
      $body += ("  - Inputs/Outputs: (to be filled)`r`n")
      $body += ("  - Week linkage: (W? tags)`r`n")
    }
    $body += "`r`n"
  }

  $weeksBlock = "## Week Status (W0-W44)`r`n`r`n| Week | Status | Expected artifacts |`r`n|---|---|---|`r`n"
  foreach($w in $weekRows){
    $weeksBlock += ("| {0} | {1} | {2} |`r`n" -f $w.week, $w.status, $w.expected.Replace('|','\|'))
  }

  $govBlock = "## Governance / Hygiene`r`n`r`n| Artifact | Exists |`r`n|---|---|`r`n"
  foreach($g in $govRows){
    $govBlock += ("| {0} | {1} |`r`n" -f $g.artifact, $g.exists)
  }

  ($hdr + $weeksBlock + "`r`n" + $govBlock + "`r`n" + $body) |
    Set-Content -LiteralPath $anchorMd -Encoding UTF8
}

Write-Host "Wrote:" -ForegroundColor Cyan
Write-Host "  $auditScriptsCsv"         -ForegroundColor Cyan
Write-Host "  $auditWeeksCsv"           -ForegroundColor Cyan
Write-Host "  $auditSummary"            -ForegroundColor Cyan
Write-Host "  $auditScriptsWithWeekCsv" -ForegroundColor Cyan
if($WriteAnchor){
  Write-Host "  $anchorMd" -ForegroundColor Cyan
}
# ===================== END =====================

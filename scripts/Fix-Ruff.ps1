# Fix-Ruff.ps1 — minimal, safe fixes for current ruff errors
# - E402: add a file-top pragma in compare_runs*.py
# - UP038: add noqa on specific isinstance() lines
# - F841: remove/underscore truly unused locals
# - E722: replace bare except with 'except Exception'
# - F401 in metrics/__init__.py: add file-level pragma (re-exports)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Add-FileTopPragma {
  param([string]$Path, [string]$Pragma)
  if (-not (Test-Path $Path)) { return }
  $text = Get-Content -Raw -Encoding UTF8 -LiteralPath $Path
  if ($text -notmatch [regex]::Escape($Pragma)) {
    # Respect shebang or encoding cookie if present
    if ($text -match '^(#!.*\r?\n)') {
      $shebang = $Matches[1]
      $rest = $text.Substring($shebang.Length)
      $new = $shebang + $Pragma + "`r`n" + $rest
    } else {
      $new = $Pragma + "`r`n" + $text
    }
    Set-Content -LiteralPath $Path -Value $new -Encoding UTF8
    Write-Host "[OK] Added pragma to $($Path): $($Pragma)"
  }
}

function Regex-Replace-InFile {
  param([string]$Path, [string]$Pattern, [string]$Replacement)
  if (-not (Test-Path $Path)) { return }
  $raw = Get-Content -Raw -Encoding UTF8 -LiteralPath $Path
  $new = [regex]::Replace($raw, $Pattern, $Replacement)
  if ($new -ne $raw) {
    Set-Content -LiteralPath $Path -Value $new -Encoding UTF8
    Write-Host "[OK] Patched $($Path)"
  }
}

# ---- Specific fixes ----

# 1) tools/add_date_if_missing.py — remove unused variable assignment (F841)
#    Replace the entire assignment line with a pure validation call (no $1 backref).
Regex-Replace-InFile `
  -Path ".\tools\add_date_if_missing.py" `
  -Pattern "(?m)^\s*test\s*=\s*pd\.to_datetime\([^)]*\)\s*$" `
  -Replacement 'pd.to_datetime(df[c], errors="raise", utc=True)'

# 2) tools/summarize_compare.py — bare except -> except Exception (E722)
Regex-Replace-InFile `
  -Path ".\tools\summarize_compare.py" `
  -Pattern "(?m)^\s*except\s*:\s*$" `
  -Replacement "except Exception:"

# 3) compare_runs*.py — import-not-top (E402) -> file pragma
Add-FileTopPragma ".\tools\compare_runs.py" "# ruff: noqa: E402"
Add-FileTopPragma ".\tools\compare_runs_fallback.py" "# ruff: noqa: E402"

# 4) UP038 — isinstance unions -> keep code, add line-level noqa
# tools/dump_parquet_schemas.py
Regex-Replace-InFile `
  -Path ".\tools\dump_parquet_schemas.py" `
  -Pattern "(?m)^(.*isinstance\(x,\s*\(np\.generic,\)\)\s*)(.*)$" `
  -Replacement '$1# noqa: UP038'
Regex-Replace-InFile `
  -Path ".\tools\dump_parquet_schemas.py" `
  -Pattern "(?m)^(.*isinstance\(x,\s*\(list,\s*tuple,\s*set\)\)\s*)(.*)$" `
  -Replacement '$1# noqa: UP038'

# tools/run_from_config.py
Regex-Replace-InFile `
  -Path ".\tools\run_from_config.py" `
  -Pattern "(?m)^(.*isinstance\(cmd,\s*\(list,\s*tuple\)\)\s*)(.*)$" `
  -Replacement '$1# noqa: UP038'

# tradingstack/io/equity.py
Regex-Replace-InFile `
  -Path ".\tradingstack\io\equity.py" `
  -Pattern "(?m)^(.*isinstance\(df\.index,\s*\(pd\.DatetimeIndex,\s*pd\.PeriodIndex\)\)\s*)(.*)$" `
  -Replacement '$1# noqa: UP038'

# tradingstack/metrics/sharpe.py
Regex-Replace-InFile `
  -Path ".\tradingstack\metrics\sharpe.py" `
  -Pattern "(?m)^(.*isinstance\(x,\s*\(list,\s*tuple\)\)\s*)(.*)$" `
  -Replacement '$1# noqa: UP038'

# 5) Unused locals F841 -> underscore them if still needed for structure
Regex-Replace-InFile `
  -Path ".\tools\make_attribution.py" `
  -Pattern "(?m)^\s*contrib_sector\s*=" `
  -Replacement "_contrib_sector ="
Regex-Replace-InFile `
  -Path ".\tools\report_portfolio_synth_v2.py" `
  -Pattern "(?m)^\s*delta\s*=" `
  -Replacement "_delta ="

# 6) tradingstack/metrics/__init__.py — lots of F401 for re-exports
Add-FileTopPragma ".\tradingstack\metrics\__init__.py" "# ruff: noqa: F401"

Write-Host "`n[OK] Fix-Ruff.ps1 completed. Re-run your commit." -ForegroundColor Green

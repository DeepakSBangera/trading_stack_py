Param(
  [Parameter(Mandatory=$true)][string]$Header,   # e.g. "## I) API Compatibility Mapping (actual → canonical)"
  [Parameter(Mandatory=$true)][string]$Body      # raw markdown to place under the header
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$tracker = ".\docs\BUILD_TRACKER.md"
if (-not (Test-Path $tracker)) { throw "Tracker not found: $tracker" }

$md = Get-Content -Path $tracker -Raw -Encoding UTF8

# Ensure header exists; if missing, append at end
if ($md -notmatch [regex]::Escape($Header)) {
  if ($md.Length -gt 0 -and -not $md.TrimEnd().EndsWith("`n")) {
    $md += "`r`n"
  }
  $md += "`r`n$Header`r`n`r`n"
}

# Replace body from $Header down to the next "## " header (or EOF)
$pattern = "(?s)" + [regex]::Escape($Header) + "\s*\r?\n(.*?)(?=(\r?\n##\s)|\Z)"
$replacement = "$Header`r`n$Body`r`n"
$md = [regex]::Replace($md, $pattern, $replacement)

Set-Content -Path $tracker -Value $md -Encoding UTF8
Write-Host "[OK] Section updated: $Header"

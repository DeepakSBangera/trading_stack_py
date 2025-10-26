Param(
  [string]$Message
)

# Robust helper to add a dated changelog entry to docs\BUILD_TRACKER.md
# Usage:
#   pwsh .\scripts\Add-TrackerChangelog.ps1 -Message "Your note here"
# If -Message is omitted, you'll be prompted.

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$tracker = ".\docs\BUILD_TRACKER.md"
if (-not (Test-Path $tracker)) {
  throw "Tracker not found at $tracker. Create it first (docs\BUILD_TRACKER.md)."
}

if ([string]::IsNullOrWhiteSpace($Message)) {
  $Message = Read-Host -Prompt "Changelog message"
  if ([string]::IsNullOrWhiteSpace($Message)) {
    throw "Message cannot be empty."
  }
}

# Read file
$content = Get-Content -Path $tracker -Raw -Encoding UTF8

$today = Get-Date -Format "yyyy-MM-dd"
$entry = "- **$today** — $Message"

# Ensure Changelog section exists; if not, add header at end
$header = "## M) Changelog"
if ($content -notmatch [regex]::Escape($header)) {
  if ($content.Length -gt 0 -and -not $content.TrimEnd().EndsWith("`n")) {
    $content += "`r`n"
  }
  $content += "`r`n$header`r`n"
}

# Insert entry directly under the Changelog header (most-recent first)
$pattern = "(?s)^(## M\) Changelog\s*\r?\n)(.*)$"
if ($content -match $pattern) {
  $before = $Matches[1]
  $after  = $Matches[2]
  if ($after -notmatch "^\s*\S") { $after = $after.TrimStart() }
  $content = $content -replace $pattern, $before + "$entry`r`n" + $after
} else {
  $content = $content.TrimEnd() + "`r`n$entry`r`n"
}

Set-Content -Path $tracker -Value $content -Encoding UTF8
Write-Host "[OK] Changelog updated: $entry"

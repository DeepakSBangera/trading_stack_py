# Restore-Files.ps1 — restore specific files to HEAD (and optionally unstage first)
Param(
  [Parameter(Mandatory=$true, ValueFromRemainingArguments=$true)]
  [string[]]$Paths,

  [switch]$UnstageFirst
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not $Paths -or $Paths.Count -eq 0) {
  throw "Usage: pwsh .\scripts\Restore-Files.ps1 <file1> <file2> ... [-UnstageFirst]"
}

# Ensure we're in the repo root
git rev-parse --show-toplevel | Out-Null

if ($UnstageFirst) {
  git restore --staged @Paths 2>$null
}

git restore --source=HEAD @Paths

Write-Host "[OK] Restored:" -ForegroundColor Green
$Paths | ForEach-Object { "  - $_" } | Write-Host

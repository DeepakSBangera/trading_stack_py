[CmdletBinding()]
param(
  [string]$Section = "W3 — Costs & Churn",
  [string]$Anchor  = "docs/ANCHOR_SCRIPTS.md",
  [string[]]$Files = @()
)

$ErrorActionPreference = "Stop"

function Resolve-Python {
  $candidates = @(".venv\Scripts\python.exe","venv\Scripts\python.exe")
  foreach($c in $candidates){ if(Test-Path $c){ return (Resolve-Path $c).Path } }
  $cmd = Get-Command python -ErrorAction SilentlyContinue
  if($cmd){ return $cmd.Source }
  throw "Python not found. Activate venv or ensure python on PATH."
}

$py = Resolve-Python

# If not provided, default to “changed scripts” in the last commit range
if(-not $Files -or $Files.Count -eq 0){
  $Files = git diff --name-only --cached
  if(-not $Files){ $Files = git diff --name-only HEAD~1..HEAD }
  $Files = $Files | Where-Object { $_ -like "scripts\*.py" }
}

if(-not $Files -or $Files.Count -eq 0){
  Write-Host "No script files detected to anchor. Pass -Files or stage changes." -ForegroundColor Yellow
  exit 0
}

$args = @("scripts/tools/make_anchor.py","--anchor",$Anchor,"--section",$Section) + $Files
& $py @args

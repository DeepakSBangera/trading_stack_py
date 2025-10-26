[CmdletBinding()]
param(
  [string]$RepoRoot = (Get-Location).Path
)

function Resolve-ConfigDir {
  param([string]$Root)
  $c1 = Join-Path $Root "configs"
  $c2 = Join-Path $Root "config"
  if (Test-Path $c1) { return "configs" }
  if (Test-Path $c2) { return "config" }
  return "(none)"
}

function Mark {
  param([bool]$Exists)
  if ($Exists) { return "OK  " } else { return "MISS" }
}

$venv   = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$tools  = Join-Path $RepoRoot "tools"
$reports= Join-Path $RepoRoot "reports"
$prices = Join-Path $RepoRoot "data_synth\prices"
$funds  = Join-Path $RepoRoot "data_synth\fundamentals"

$configDirName = Resolve-ConfigDir -Root $RepoRoot
$configDirPath = if ($configDirName -ne "(none)") { Join-Path $RepoRoot $configDirName } else { $null }

Write-Host "=== Repo Snapshot ===" -ForegroundColor Cyan
"{0,-20} {1}" -f "RepoRoot:", $RepoRoot
"{0,-20} {1} {2}" -f "Python:",  (Mark (Test-Path $venv)),   $venv
"{0,-20} {1} {2}" -f "tools:",   (Mark (Test-Path $tools)),  $tools
"{0,-20} {1} {2}" -f "reports:", (Mark (Test-Path $reports)),$reports
"{0,-20} {1} {2}" -f "data_synth\prices:", (Mark (Test-Path $prices)), $prices
"{0,-20} {1} {2}" -f "data_synth\fundamentals:", (Mark (Test-Path $funds)), $funds
"{0,-20} {1}" -f "config(s):", $configDirName

if ($configDirPath) {
  $yaml = Get-ChildItem -File -Filter *.y*ml $configDirPath -ErrorAction SilentlyContinue
  "{0,-20} {1}" -f "configs count:", ($yaml.Count)
  if ($yaml.Count -gt 0) {
    "Top config files:"
    $yaml | Select-Object -First 5 | ForEach-Object { " - {0}" -f $_.FullName }
  }
}

"Tools present:"
if (Test-Path $tools) {
  Get-ChildItem -File $tools -ErrorAction SilentlyContinue | ForEach-Object {
    " - {0}" -f $_.Name
  }
} else {
  " (tools folder missing)"
}

"Recent reports (up to 5):"
if (Test-Path $reports) {
  Get-ChildItem -File $reports | Sort-Object LastWriteTime -Descending | Select-Object -First 5 |
    ForEach-Object { "{0:yyyy-MM-dd HH:mm}  {1}" -f $_.LastWriteTime, $_.Name }
} else {
  "(no reports dir)"
}

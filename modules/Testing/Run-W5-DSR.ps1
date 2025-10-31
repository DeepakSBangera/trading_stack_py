# scripts/Run-W5-DSR.ps1
[CmdletBinding()]
param(
  [string]$PythonPath = ".\.venv\Scripts\python.exe",
  [string]$OutParquet = ".\reports\wk5_walkforward_dsr.parquet",
  [string]$DecisionNote = ".\reports\promotion_decision.txt",
  [int]$NTrials = 10,
  [double]$MinDSR = 0.00,
  [double]$MaxPBO = 0.20,
  [string[]]$Candidates = @(".\reports\wk5_walkforward.parquet", ".\reports\portfolio_v2.parquet")
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $PythonPath)) {
  throw "Venv Python not found at $PythonPath"
}

$tool = ".\tools\dsr_pbo.py"
if (-not (Test-Path $tool)) {
  throw "Missing tool: $tool"
}

# Ensure out dir exists
$dir = Split-Path -Parent $OutParquet
if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }

Write-Host "Running W5 DSR/PBO..."
# Build args: dsr_pbo.py --out ... --note ... --n-trials ... --min-dsr ... --max-pbo ... CANDIDATES...
$argv = @($tool, "--out", $OutParquet, "--note", $DecisionNote, "--n-trials", $NTrials, "--min-dsr", $MinDSR, "--max-pbo", $MaxPBO) + $Candidates

& $PythonPath @argv
if ($LASTEXITCODE -ne 0) { throw "dsr_pbo.py failed with exit code $LASTEXITCODE" }

Write-Host "W5 DSR complete -> $OutParquet"
Write-Host "Decision note    -> $DecisionNote"

# Open the decision note for quick review
if (Test-Path $DecisionNote) {
  Start-Process notepad.exe $DecisionNote
}

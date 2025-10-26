[CmdletBinding()]
param(
  [string]$Config = "config\baseline.json"
)
$ErrorActionPreference = "Stop"
$py = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "venv python not found at $py" }

Write-Host "Running pipeline from config: $Config" -ForegroundColor Cyan
& $py .\tools\run_from_config.py $Config

# --- Post step: W5 DSR/PBO ---
try {
  Write-Host "Post-step: DSR/PBO..."
  powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\Run-DSR.ps1"
} catch {
  Write-Warning "DSR/PBO step failed: $($_.Exception.Message)"
}
# --- End post step ---


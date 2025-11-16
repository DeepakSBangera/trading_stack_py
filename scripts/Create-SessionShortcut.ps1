# scripts/Create-SessionShortcut.ps1
# Creates a desktop shortcut to run the full pipeline (run_session_SOP.cmd)

$ErrorActionPreference = 'Stop'

# --- Paths ---
$ROOT = 'F:\Projects\trading_stack_py'
$CmdPath = Join-Path $ROOT 'scripts\run_session_SOP.cmd'
$WorkDir = $ROOT
$Desktop = [Environment]::GetFolderPath('Desktop')
$LnkPath = Join-Path $Desktop 'Trading Session SOP.lnk'

# --- Icon logic: prefer venv Python icon, else shell32.dll fallback ---
$VenvPy = Join-Path $ROOT '.venv\Scripts\python.exe'
$IconLocation = if (Test-Path $VenvPy) { $VenvPy } else { "$env:SystemRoot\System32\shell32.dll,1" }

# --- Validate required files ---
if (!(Test-Path $CmdPath)) {
  throw "Missing orchestrator CMD: $CmdPath`nCreate it first (run_session_SOP.cmd)."
}

# --- Create the shortcut via WScript.Shell COM ---
$wsh = New-Object -ComObject WScript.Shell
$sc = $wsh.CreateShortcut($LnkPath)
$sc.TargetPath = $CmdPath
$sc.WorkingDirectory = $WorkDir
$sc.IconLocation = $IconLocation
$sc.WindowStyle = 1          # Normal window
$sc.Description = 'Run W7→W8→W11→W12 pipeline (no popups; manifest at end)'
$null = $sc.Save()

# --- Output & open Desktop to show the shortcut ---
Write-Host "Created shortcut:" -ForegroundColor Green
Write-Host "  $LnkPath"
Write-Host "Target:" (Get-Item $CmdPath).FullName
Write-Host "Icon:  $IconLocation"
Start-Sleep -Milliseconds 200
Start-Process explorer.exe $Desktop


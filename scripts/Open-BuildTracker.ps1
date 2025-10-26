# Open the living tracker
Set-Location F:\Projects\trading_stack_py
if (-not (Test-Path .\docs\BUILD_TRACKER.md)) { New-Item -ItemType File .\docs\BUILD_TRACKER.md | Out-Null }
notepad .\docs\BUILD_TRACKER.md

$ErrorActionPreference = "Stop"
.\scripts\Toggle-DataSource.ps1 -Mode kite
.\.venv\Scripts\python.exe .\tools\kite_cache_stub.py
Write-Host "Kite scaffold verified (still no API spend)." -ForegroundColor Green

@echo off
setlocal
REM -----------------------------------------------------------------------------
REM  run_session_SOP.cmd  —  Double-click to run the full pipeline (W7→W8→W11→W12)
REM  - Uses your venv python
REM  - Disables auto-open during steps to avoid file locks
REM  - Writes one manifest at the end, builds W12 review ZIP, appends tracker line
REM -----------------------------------------------------------------------------

set "ROOT=F:\Projects\trading_stack_py"
set "PY=%ROOT%\.venv\Scripts\python.exe"
set "ORCH=%ROOT%\scripts\run_session_SOP.py"

REM Make sure we don’t pop Notepad/Excel during steps
set "RUN_INFO_OPEN=none"
set "RUN_INFO_MANIFEST=on"

cd /d "%ROOT%"

if not exist "%PY%" (
  echo [ERROR] Python venv not found: %PY%
  echo Open PowerShell and run:
  echo   python -m venv .venv ^&^& .\.venv\Scripts\python.exe -m pip install -U pip
  echo Then install your deps and re-run this CMD.
  echo.
  pause
  exit /b 1
)

if not exist "%ORCH%" (
  echo [ERROR] Orchestrator missing: %ORCH%
  echo Create it with notepad .\scripts\run_session_SOP.py and paste the full drop-in.
  echo.
  pause
  exit /b 1
)

echo === Session SOP start ===
"%PY%" "%ORCH%"
set RC=%ERRORLEVEL%

echo.
if %RC%==0 (
  echo === Pipeline finished OK (rc=%RC%) ===
) else (
  echo === Pipeline finished with errors (rc=%RC%) ===
)

echo.
echo Press any key to close...
pause >nul
exit /b %RC%

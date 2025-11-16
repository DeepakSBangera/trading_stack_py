[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'

# Resolve python from venv, fallback to PATH
$VenvPy = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
if (-not (Test-Path $VenvPy)) { $VenvPy = "python" }

function Run-Py([string]$code) {
  & $VenvPy -c $code
  if ($LASTEXITCODE) { throw "Python snippet failed" }
}

Write-Host "[1/2] Import surface…" -ForegroundColor Cyan
Run-Py @"
import tradingstack
import tradingstack.metrics as m
import tradingstack.factors as f
print('OK: imports')
print('sharpe_daily:', hasattr(m,'sharpe_daily'))
print('momentum:', hasattr(f,'momentum_12_1_proxy'))
"@

Write-Host "[2/2] Sanity numbers…" -ForegroundColor Cyan
Run-Py @"
import numpy as np
from tradingstack.metrics import sharpe_daily, sharpe_annual
r=[0.01,-0.005,0.006,0.004,-0.002]
print('sharpe_daily=', round(sharpe_daily(r),4))
print('sharpe_annual=', round(sharpe_annual(r),4))
"@

Write-Host "[OK] Kit looks healthy." -ForegroundColor Green

Param(
  [string]$ConfigPath = "config/rolling.json"
)

# Ensure config exists (create sensible default if missing)
if (-not (Test-Path -Path "config")) { New-Item -ItemType Directory -Path "config" | Out-Null }
if (-not (Test-Path -Path $ConfigPath)) {
@"
{
  "nav_file": "reports/portfolio_v2.parquet",
  "nav_col": "_nav",
  "ret_window_sharpe": 252,
  "ret_window_sortino": 252,
  "vol_window": 63,
  "dd_window": 252,
  "annualization": 252,
  "rf_per_period": 0.0,
  "regime_method": "sma_crossover",
  "regime_fast": 50,
  "regime_slow": 200,
  "out_parquet": "reports/rolling_metrics.parquet",
  "out_summary": "reports/rolling_metrics_summary.txt"
}
"@ | Set-Content -Encoding UTF8 $ConfigPath
  Write-Host "[INIT] Created default $ConfigPath"
}

$py = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) { Write-Error "Python venv not found at $py"; exit 1 }

& $py "tools\make_rolling_metrics.py" $ConfigPath

if (Test-Path "reports\rolling_metrics_summary.txt") {
  Write-Host "`n===== SUMMARY ====="
  Get-Content "reports\rolling_metrics_summary.txt"
} else {
  Write-Warning "Summary file not found."
}

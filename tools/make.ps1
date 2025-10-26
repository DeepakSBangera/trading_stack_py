# tools/make.ps1  — lightweight task runner for this repo

function _InRepoRoot {
  if (-not (Test-Path -Path .git)) {
    throw "Run this from the repo root (folder containing .git). Current: $(Get-Location)"
  }
}

function _Req($exe) {
  if (-not (Get-Command $exe -ErrorAction SilentlyContinue)) {
    throw "Required tool '$exe' not found in PATH."
  }
}

function Use-Venv {
  if (-not $env:VIRTUAL_ENV) {
    Write-Warning "No venv active. Activate first: .\.venv\Scripts\Activate.ps1"
  } else {
    Write-Host "Using venv: $env:VIRTUAL_ENV"
  }
}

function Clean {
  _InRepoRoot
  Write-Host "Cleaning build artifacts..."
  Remove-Item -Recurse -Force dist, build -ErrorAction SilentlyContinue
  Get-ChildItem -Recurse -Include *.egg-info | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
  Write-Host "Done."
}

function Build {
  _InRepoRoot; Use-Venv; _Req python
  Write-Host "Building sdist+wheel..."
  python -m pip install -U build | Out-Host
  python -m build | Out-Host
}

function TwineCheck {
  _InRepoRoot; Use-Venv; _Req python
  python -m pip install -U twine | Out-Host
  twine check dist/* | Out-Host
}

function Install-LatestWheel {
  _InRepoRoot; Use-Venv; _Req python
  $whl = Get-ChildItem dist\*.whl | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  if (-not $whl) { throw "No wheel in dist/. Run Build first." }
  Write-Host "Installing $($whl.Name) ..."
  pip install --upgrade $whl.FullName | Out-Host
}

function Smoke-Run {
  _InRepoRoot; Use-Venv; _Req trading-stack
  trading-stack --ticker RELIANCE.NS --start 2015-01-01 --source synthetic | Out-Host
}

function Smoke-Portfolio {
  _InRepoRoot; Use-Venv; _Req trading-stack-portfolio
  trading-stack-portfolio --tickers RELIANCE.NS,HDFCBANK.NS,INFY.NS,ICICIBANK.NS,TCS.NS `
    --start 2018-01-01 --source synthetic --lookback 126 --top_n 3 --cost_bps 10 | Out-Host
}

function Show-PackageVersion {
  _InRepoRoot; Use-Venv; _Req python
  python -c "import importlib.metadata as im; print(im.version('trading-stack-py'))"
}

function Bump-Version {
  param([Parameter(Mandatory)][string]$New)
  _InRepoRoot
  $raw = Get-Content pyproject.toml -Raw
  $out = $raw -replace 'version\s*=\s*"\d+\.\d+\.\d+"', "version = `"$New`""
  Set-Content -Encoding UTF8 -NoNewline pyproject.toml -Value $out
  git add pyproject.toml
  git commit -m "bump: $New" | Out-Host
  git tag -a "v$New" -m "Release $New" | Out-Host
  git push --follow-tags | Out-Host
  Write-Host "Version bumped to $New and tag pushed."
}

function All {
  Clean
  Build
  TwineCheck
  Install-LatestWheel
  Smoke-Run
  Smoke-Portfolio
  Show-PackageVersion
  Write-Host "All steps done."
}
function Ensure-PlotDeps {
  Use-Venv; _Req python
  Write-Host "Ensuring plotting deps (matplotlib, pandas) ..."
  python -m pip install -U matplotlib pandas | Out-Host
}

function Report-Run {
  param(
    [string]$Ticker,              # e.g. RELIANCE.NS
    [string]$CsvPath              # optional explicit CSV path
  )
  _InRepoRoot; Use-Venv; _Req python
  Ensure-PlotDeps

  if (-not $CsvPath) {
    if (-not $Ticker) { throw "Provide -Ticker or -CsvPath" }
    # run CSV pattern: reports\run_RELIANCE_NS.csv (dots => underscores)
    $safe = $Ticker -replace '\W','_'
    $CsvPath = Join-Path "reports" "run_$($safe).csv"
  }
  if (-not (Test-Path $CsvPath)) { throw "CSV not found: $CsvPath" }

  python tools/plot_report.py --csv "$CsvPath" --outdir "reports" --title "$Ticker" | Out-Host
}

function Report-Portfolio {
  param([string]$CsvPath)
  _InRepoRoot; Use-Venv; _Req python
  Ensure-PlotDeps
  if (-not $CsvPath) {
    # try to pick the most recent portfolio metrics file
    $latest = Get-ChildItem reports\portfolio_*_metrics.csv -ErrorAction SilentlyContinue |
      Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $latest) { throw "No portfolio_*_metrics.csv found. Provide -CsvPath." }
    $CsvPath = $latest.FullName
  }
  # Reuse plot_report by treating an equity-like column if present
  python tools/plot_report.py --csv "$CsvPath" --outdir "reports" --title "Portfolio" | Out-Host
}

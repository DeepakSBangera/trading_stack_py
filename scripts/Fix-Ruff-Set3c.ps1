# Fix-Ruff-Set3c.ps1 — add `import matplotlib as mpl` before mpl.use("Agg")
$ErrorActionPreference = 'Stop'

function Patch-File($Path) {
  if (-not (Test-Path $Path)) { throw "Missing: $Path" }
  $t = Get-Content -Raw -LiteralPath $Path

  # If there's a plain `import matplotlib`, convert it to alias form.
  $t2 = [regex]::Replace($t, '^(?m)\s*import\s+matplotlib\s*$', 'import matplotlib as mpl')

  # If alias still not present, insert it just before mpl.use("Agg")
  if ($t2 -notmatch '(?m)^\s*import\s+matplotlib\s+as\s+mpl\s*$') {
    $t2 = [regex]::Replace(
      $t2,
      '^(?m)(\s*mpl\.use\("Agg"\)\s*)$',
      "import matplotlib as mpl`r`n`$1"
    )
  }

  if ($t2 -ne $t) {
    Set-Content -LiteralPath $Path -Value $t2 -NoNewline
    Write-Host "[OK] Patched $Path"
  } else {
    Write-Host "[..] No change $Path"
  }
}

$targets = @(
  'tools\pilot_backtest.py',
  'tools\report_portfolio_v2.py'
)

foreach ($f in $targets) { Patch-File $f }

Write-Host "`n[OK] Fix-Ruff-Set3c.ps1 completed."

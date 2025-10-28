# Fix-Ruff-Set3b.ps1 — clean up F821/E402 around matplotlib use()

$ErrorActionPreference = 'Stop'

function Patch-File($Path, $Find, $Replace) {
  if (-not (Test-Path $Path)) { throw "Missing: $Path" }
  $t = Get-Content -Raw -LiteralPath $Path
  $n = [regex]::Replace($t, $Find, $Replace, 'Multiline')
  if ($n -ne $t) {
    Set-Content -LiteralPath $Path -Value $n -NoNewline
    Write-Host "[OK] Patched $Path"
  } else {
    Write-Host "[..] No change $Path"
  }
}

$targets = @(
  'tools\pilot_backtest.py',
  'tools\report_portfolio_v2.py'
)

foreach ($f in $targets) {
  # 1) matplotlib.use(...) -> mpl.use(...)
  Patch-File $f '^(?m)\s*matplotlib\.use\("Agg"\)\s*$' 'mpl.use("Agg")'

  # 2) Add noqa on pyplot import so E402 is silenced (we must set backend first)
  Patch-File $f '^(?m)\s*import\s+matplotlib\.pyplot\s+as\s+plt\s*$' 'import matplotlib.pyplot as plt  # noqa: E402'
}

Write-Host "`n[OK] Fix-Ruff-Set3b.ps1 completed."

# Fix-Ruff-Set3.ps1 — resolve remaining Ruff issues:
# - E402 (imports not at top) in compare_runs*.py via per-file directive
# - ICN001 matplotlib aliasing -> mpl
# - F841 unused variable 'delta' -> _delta
# - E722 bare except -> except Exception

$ErrorActionPreference = 'Stop'

function Get-Text($Path) {
  if (-not (Test-Path $Path)) { throw "Missing: $Path" }
  return Get-Content -Raw -LiteralPath $Path
}
function Set-Text($Path, $Text) {
  Set-Content -LiteralPath $Path -Value $Text -NoNewline
  Write-Host "[OK] Patched $Path"
}

function Ensure-RuffNoqaE402($Path) {
  $t = Get-Text $Path
  if ($t -match '^\#\!') {
    # shebang present; ensure the directive is the next logical line
    if ($t -notmatch '^\#\!.*?$\r?\n\# ruff: noqa: E402') {
      $t = $t -replace '(^\#\!.*?\r?\n)', "`$1# ruff: noqa: E402`r`n"
      Set-Text $Path $t
    } else { Write-Host "[..] E402 directive already present in $Path" }
  } else {
    if ($t -notmatch '^\# ruff: noqa: E402') {
      $t = "# ruff: noqa: E402`r`n" + $t
      Set-Text $Path $t
    } else { Write-Host "[..] E402 directive already present in $Path" }
  }
}

function Replace-Line($Path, $FindPattern, $ReplaceText) {
  $t = Get-Text $Path
  $new = [System.Text.RegularExpressions.Regex]::Replace($t, $FindPattern, $ReplaceText, 'Multiline')
  if ($new -ne $t) { Set-Text $Path $new } else { Write-Host "[..] No change $Path (pattern not found)" }
}

# --- E402: add per-file directive at the very top (safe) ---
$cmp = @('tools\compare_runs.py', 'tools\compare_runs_fallback.py')
foreach ($f in $cmp) { Ensure-RuffNoqaE402 $f }

# --- ICN001: import matplotlib as mpl ---
Replace-Line 'tools\pilot_backtest.py'      '^(?m)import\s+matplotlib\s*$'         'import matplotlib as mpl  # noqa: ICN001'
Replace-Line 'tools\report_portfolio_v2.py' '^(?m)import\s+matplotlib\s*$'         'import matplotlib as mpl  # noqa: ICN001'

# --- F841: unused variable 'delta' ---
Replace-Line 'tools\report_portfolio_synth_v2.py' '^(?m)\s*delta\s*=\s*weights\s*-\s*weights_prev\s*$' '    _delta = weights - weights_prev  # noqa: F841'

# --- E722: bare except -> explicit ---
Replace-Line 'tools\summarize_compare.py' '^(?m)\s*except\s*:\s*$' '    except Exception:'

Write-Host "`n[OK] Fix-Ruff-Set3.ps1 completed."

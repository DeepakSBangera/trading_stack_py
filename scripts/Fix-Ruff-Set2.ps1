# Fix-Ruff-Set2.ps1 — minimal, safe text edits for current Ruff findings.

$ErrorActionPreference = 'Stop'

function Patch-Line {
  param(
    [Parameter(Mandatory)] [string]$Path,
    [Parameter(Mandatory)] [string]$Find,
    [Parameter(Mandatory)] [string]$Replace
  )
  if (-not (Test-Path $Path)) { Write-Warning "Missing: $Path"; return }
  $txt = Get-Content -Raw -LiteralPath $Path
  $new = $txt -replace $Find, $Replace
  if ($new -ne $txt) {
    Set-Content -LiteralPath $Path -Value $new -NoNewline
    Write-Host "[OK] Patched $Path"
  } else {
    Write-Host "[..] No change $Path (pattern not found)"
  }
}

# --- B007: unused loop vars ---
Patch-Line -Path 'scripts\check_structure.py' `
  -Find 'for root, dirs, files in os\.walk\("\.", topdown=True\):' `
  -Replace 'for root, dirs, _files in os.walk(".", topdown=True):'

Patch-Line -Path 'scripts\fetch_prices.py' `
  -Find 'for k in range\(\s*1,\s*attempts \+ 1\s*\):' `
  -Replace 'for _k in range(1, attempts + 1):'

Patch-Line -Path 'scripts\run_w5_demo.py' `
  -Find 'for tr,\s*te in cv\.split\(n\):' `
  -Replace 'for _tr, te in cv.split(n):'

Patch-Line -Path 'tests\test_walkforward.py' `
  -Find 'for tr,\s*te in splits:' `
  -Replace 'for _tr, te in splits:'

# --- F841 / UP/C/N/E rules (surgical) ---

# tools\add_date_if_missing.py: drop the unused 'test =' assignment
Patch-Line -Path 'tools\add_date_if_missing.py' `
  -Find 'test\s*=\s*pd\.to_datetime\(df\[c\],\s*errors="raise",\s*utc=True\)' `
  -Replace 'pd.to_datetime(df[c], errors="raise", utc=True)  # validate only'

# tools\build_portfolio.py: C416 dict comprehension -> dict()
Patch-Line -Path 'tools\build_portfolio.py' `
  -Find 'pd\.DataFrame\(\{dt:\s*w\s*for\s*dt,\s*w\s*in\s*weights\}\)\.T' `
  -Replace 'pd.DataFrame(dict(weights)).T'

# compare_runs*: add noqa to imports to silence E402 (imports must remain where they are)
$cmpFiles = @('tools\compare_runs.py','tools\compare_runs_fallback.py')
$importFixes = @(
  @{ F='^import argparse\s*$'; R='import argparse  # noqa: E402' },
  @{ F='^import os\s*$'; R='import os  # noqa: E402' },
  @{ F='^import numpy as np\s*$'; R='import numpy as np  # noqa: E402' },
  @{ F='^import pandas as pd\s*$'; R='import pandas as pd  # noqa: E402' },
  @{ F='^from tradingstack\.io import load_equity\s*$'; R='from tradingstack.io import load_equity  # noqa: E402' },
  @{ F='^from tradingstack\.metrics import sharpe_annual\s*$'; R='from tradingstack.metrics import sharpe_annual  # noqa: E402' }
)
foreach ($f in $cmpFiles) {
  foreach ($fix in $importFixes) {
    Patch-Line -Path $f -Find $fix.F -Replace $fix.R
  }
}

# tools\make_attribution.py: F841 — unused variable
Patch-Line -Path 'tools\make_attribution.py' `
  -Find '\bcontrib_sector\s*=' `
  -Replace '_contrib_sector = '

# ICN001: import matplotlib as mpl
Patch-Line -Path 'tools\pilot_backtest.py' `
  -Find '^import matplotlib\s*$' `
  -Replace 'import matplotlib as mpl  # noqa: ICN001'
Patch-Line -Path 'tools\report_portfolio_v2.py' `
  -Find '^import matplotlib\s*$' `
  -Replace 'import matplotlib as mpl  # noqa: ICN001'

# tools\report_portfolio_synth_v2.py: F841 — unused variable
Patch-Line -Path 'tools\report_portfolio_synth_v2.py' `
  -Find '^\s*delta\s*=\s*weights\s*-\s*weights_prev\s*$' `
  -Replace '    _delta = weights - weights_prev  # intentional unused'

# tools\summarize_compare.py: E722 — bare except
Patch-Line -Path 'tools\summarize_compare.py' `
  -Find '^\s*except:\s*$' `
  -Replace '    except Exception:'

Write-Host "`n[OK] Fix-Ruff-Set2.ps1 completed. Now run pre-commit."

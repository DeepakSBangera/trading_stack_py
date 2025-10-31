[CmdletBinding()]
param(
  [string] $RepoRoot = (Resolve-Path ".").Path
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$RepoRoot = (Resolve-Path $RepoRoot).Path
Set-Location $RepoRoot

$wantDirs = @(
  'app','config','docs','pipelines','pricing','econo',
  'reports','scripts','src','tests','tools','tradingstack'
)
foreach ($d in $wantDirs) {
  $p = Join-Path $RepoRoot $d
  if (-not (Test-Path $p)) { New-Item -ItemType Directory -Force -Path $p | Out-Null }
}

$routes = @(
  @{ Pattern = '^Make-.*\.ps1$';        Dest = 'scripts' },
  @{ Pattern = '^View-.*\.ps1$';        Dest = 'scripts' },
  @{ Pattern = '^Quick-.*\.ps1$';       Dest = 'scripts' },
  @{ Pattern = '^Fix-.*\.ps1$';         Dest = 'scripts' },
  @{ Pattern = '.*\.ps1$';              Dest = 'scripts' },
  @{ Pattern = '.*\.py$';               Dest = 'tools' },
  @{ Pattern = '^README\.md$';          Dest = '' },
  @{ Pattern = '^requirements.*\.txt$'; Dest = '' },
  @{ Pattern = '^pyproject\.toml$';     Dest = '' },
  @{ Pattern = '^\.pre-commit-config\.yaml$'; Dest = '' },
  @{ Pattern = '^\.gitattributes$';     Dest = '' },
  @{ Pattern = '^_make_toy_w6\.py$';    Dest = 'tools' }
)

$rootItems = Get-ChildItem -LiteralPath $RepoRoot -File -Force `
  | Where-Object { $_.Name -notmatch '^(\.git|\.venv|data(|_live|_synth))$' }

foreach ($it in $rootItems) {
  $moved = $false
  foreach ($r in $routes) {
    if ($it.Name -match $r.Pattern) {
      $dest = if ($r.Dest -ne '') { Join-Path $RepoRoot $r.Dest } else { $RepoRoot }
      if (-not (Test-Path $dest)) { New-Item -ItemType Directory -Force -Path $dest | Out-Null }
      $target = Join-Path $dest $it.Name
      if ($target -ne $it.FullName) { Move-Item -LiteralPath $it.FullName -Destination $target -Force }
      $moved = $true; break
    }
  }
}

Write-Host "[OK] Structure enforced." -ForegroundColor Green

param(
  [string]$Task = "regression",            # or "classification"
  [string]$TagW8 = "REL_W8",
  [string]$TagW7 = "REL_W7",
  [string]$TagW9 = "REL_W9",
  [string]$W6Dir = ""                      # leave blank to auto-pick newest
)

$ErrorActionPreference = "Stop"

function Get-NewestDir($root) {
  if (-not (Test-Path $root)) { throw "Path not found: $root" }
  $d = Get-ChildItem $root -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  if (-not $d) { throw "No subfolders inside $root" }
  return $d.FullName
}

# 1) Resolve W6
if ([string]::IsNullOrWhiteSpace($W6Dir)) {
  $W6Dir = Get-NewestDir "reports\W6"
}
Write-Host "Using W6:" $W6Dir

# 2) Run W8 (tuning)
python -m trading_stack_py.pipelines.tune_models `
  --w6-dir "$W6Dir" `
  --task $Task `
  --tag $TagW8 `
  --outdir reports/W8

$W8 = Get-NewestDir "reports\W8"
Write-Host "W8:" $W8

# 3) Run W7 (training with chosen params; safe to re-run)
python -m trading_stack_py.pipelines.train_models `
  --w6-dir "$W6Dir" `
  --task $Task `
  --tag $TagW7 `
  --outdir reports/W7

$W7 = Get-NewestDir "reports\W7"
Write-Host "W7:" $W7

# 4) Run W9 (evaluation/join)
python -m trading_stack_py.pipelines.evaluate_models `
  --w6-dir "$W6Dir" `
  --w7-dir "$W7" `
  --tag $TagW9 `
  --outdir reports/W9

$W9 = Get-NewestDir "reports\W9"
Write-Host "W9:" $W9

# Open summaries (VS Code)
if (Get-Command code -ErrorAction SilentlyContinue) {
  code "$W8\README.md"
  code "$W7\README.md"
  code "$W9\README.md"
}

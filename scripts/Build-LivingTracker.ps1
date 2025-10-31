[CmdletBinding()]
param([string] $RepoRoot = (Resolve-Path ".").Path)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$RepoRoot = (Resolve-Path $RepoRoot).Path
$docs = Join-Path $RepoRoot 'docs'
if (-not (Test-Path $docs)) { New-Item -ItemType Directory -Force -Path $docs | Out-Null }

$csvOut = Join-Path $docs 'living_tracker.csv'
$mdOut  = Join-Path $docs 'living_tracker.md'

$includeExt = @('.py','.ps1','.md','.toml','.yaml','.yml','.json','.csv','.html','.png','.txt')
$files = Get-ChildItem -LiteralPath $RepoRoot -Recurse -Force -File -ErrorAction SilentlyContinue `
  | Where-Object {
      ($includeExt -contains $_.Extension.ToLower()) -and
      ($_.FullName -notmatch '\\\.git\\|\\\.venv\\|\\__pycache__\\|\\_cache\\|\\_tmp\\')
    }

$gitTracked = @{}; $gitUntracked = @{}
try {
  $tracked = git ls-files | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne '' }
  foreach ($t in $tracked) { $gitTracked[$t.ToLower()] = $true }
  $untracked = git ls-files --others --exclude-standard | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne '' }
  foreach ($u in $untracked) { $gitUntracked[$u.ToLower()] = $true }
} catch {}

"path,week,label,size,last_write,status,next_action" | Set-Content -LiteralPath $csvOut -Encoding UTF8

function Get-WeekLabel([string]$rel) {
  if ($rel -match 'reports[/\\](wk\d+)_([^/\\]+)\.(csv|html|png|txt)$') { return @($matches[1], $matches[2]) }
  return @('','')
}

foreach ($f in $files) {
  $rel = $f.FullName.Substring($RepoRoot.Length).TrimStart('\','/')
  $k = $rel.ToLower()
  $status = if ($gitTracked.ContainsKey($k)) { 'tracked' } elseif ($gitUntracked.ContainsKey($k)) { 'untracked' } else { 'unknown' }
  $wl = Get-WeekLabel $rel
  $week = $wl[0]; $label= $wl[1]
  Add-Content -LiteralPath $csvOut -Value ('"{0}",{1},{2},{3},"{4}",{5},' -f $rel,$week,$label,$f.Length,($f.LastWriteTime.ToString('yyyy-MM-dd HH:mm')), $status)
}

$rows = Import-Csv -LiteralPath $csvOut
$groups = $rows | Group-Object week
"# Living Tracker (single source of truth)

- Generated: $(Get-Date -Format 'yyyy-MM-dd HH:mm')
- Repo: $RepoRoot

" | Set-Content -LiteralPath $mdOut -Encoding UTF8

foreach ($g in $groups) {
  $wk = if ($g.Name) { $g.Name } else { '(no week tag)' }
  Add-Content -LiteralPath $mdOut -Value "## $wk`r`n"
  Add-Content -LiteralPath $mdOut -Value "| label | path | size | last_write | status |"
  Add-Content -LiteralPath $mdOut -Value "|---|---|---:|---|---|"
  $g.Group | Sort-Object label, path | ForEach-Object {
    Add-Content -LiteralPath $mdOut -Value ("| {0} | {1} | {2} | {3} | {4} |" -f $_.label, $_.path, $_.size, $_.last_write, $_.status)
  }
  Add-Content -LiteralPath $mdOut -Value "`r`n"
}

Write-Host "[OK] Living tracker written to:" -ForegroundColor Green
Write-Host " - $csvOut"
Write-Host " - $mdOut"

<#
.SYNOPSIS
  Create/push an annotated Git tag for a release and (optionally) push the current branch.

.PARAMETER Tag
  The Git tag to create/push (e.g., v1.0, v1.0.1).

.PARAMETER Message
  Annotated tag message.

.PARAMETER Force
  Skip the dirty working-tree guard.

.PARAMETER PushBranch
  Also push the current branch (and set upstream if missing). Default: true.
#>

[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)][string]$Tag,
  [Parameter(Mandatory = $true)][string]$Message,
  [switch]$Force,
  [bool]$PushBranch = $true
)

$ErrorActionPreference = "Stop"

function Fail($msg) { Write-Error $msg; exit 1 }

# 0) Ensure Git is available
try { git --version *>$null } catch { Fail "Git not found in PATH." }

# 1) Ensure we’re in a git repo
try { git rev-parse --git-dir *>$null } catch { Fail "Not inside a git repository." }

# 2) Working tree clean (unless -Force)
$porcelain = git status --porcelain
if ($porcelain -and -not $Force) {
  Fail "Working tree is not clean. Commit/stash or pass -Force to override."
}

# 3) Info
$branch = (git rev-parse --abbrev-ref HEAD).Trim()
$last = git log -1 --pretty=format:"%h | %ci | %s"
Write-Host "Branch : $branch"
Write-Host "Last   : $last"

# 4) Create tag if it doesn't exist
$exists = (git tag --list $Tag)
if (-not $exists) {
  git tag -a $Tag -m $Message
  Write-Host "Created tag $Tag"
} else {
  Write-Host "Tag $Tag already exists (skipping create)."
}

# 5) Push branch (if requested) and set upstream when needed
if ($PushBranch) {
  try {
    git rev-parse --abbrev-ref --symbolic-full-name "@{u}" *> $null
    Write-Host "Upstream is set for '$branch' (skip push --set-upstream)."
  } catch {
    Write-Host "Setting upstream for '$branch'…"
    git push --set-upstream origin $branch
  }
}

# 6) Push tag
git push origin $Tag
Write-Host "Pushed tag $Tag to origin."

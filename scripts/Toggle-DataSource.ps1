[CmdletBinding()]
param(
  [ValidateSet('synth','kite')]
  [string]$Mode = 'synth',

  [switch]$EnableKite,

  # Path to data_source.json
  [string]$ConfigPath = 'config\data_source.json',

  # If you truly need UTF-8 without BOM, pass -NoBOM
  [switch]$NoBOM
)

$ErrorActionPreference = 'Stop'

# Ensure config folder exists
$cfgDir = Split-Path -Parent $ConfigPath
if (-not (Test-Path $cfgDir)) { New-Item -ItemType Directory -Force -Path $cfgDir | Out-Null }

# Load existing config (tolerate BOM) or create defaults
$cfg = $null
if (Test-Path $ConfigPath) {
  try {
    $raw = Get-Content -Raw -Path $ConfigPath
    $cfg = $raw | ConvertFrom-Json
  } catch {
    Write-Warning "Existing $ConfigPath not valid JSON; recreating with defaults."
  }
}

if (-not $cfg) {
  $cfg = [pscustomobject]@{
    mode  = 'synth'
    synth = [pscustomobject]@{
      prices_root       = 'data_synth/prices'
      fundamentals_root = 'data_synth/fundamentals'
    }
    kite  = [pscustomobject]@{
      enabled            = $false
      dry_run            = $true
      cache_root         = 'data_live'
      rate_limit_per_min = 90
      backoff_seconds    = 2.0
      universe_file      = 'config/universe_kite.csv'
      pit_log            = 'data_live/_pit/log.jsonl'
    }
  }
}

# Apply requested changes
$cfg.mode = $Mode
if ($Mode -eq 'synth') {
  $cfg.kite.enabled = $false
} else {
  $cfg.kite.enabled = [bool]$EnableKite
}

# Create minimal directory scaffolding when kite is selected
if ($Mode -eq 'kite') {
  $roots = @(
    $cfg.kite.cache_root,
    (Join-Path $cfg.kite.cache_root '_pit'),
    (Split-Path $cfg.kite.universe_file -Parent)
  )
  foreach ($r in $roots) {
    if ($r -and -not (Test-Path $r)) { New-Item -ItemType Directory -Force -Path $r | Out-Null }
  }
}

# Serialize
$json = $cfg | ConvertTo-Json -Depth 8

# Write with correct encoding
if ($NoBOM) {
  $enc = New-Object System.Text.UTF8Encoding($false)
  $full = Resolve-Path $ConfigPath
  [System.IO.File]::WriteAllText($full, $json, $enc)
} else {
  Set-Content -Path $ConfigPath -Value $json -Encoding UTF8
}

Write-Host ("Data source set to '{0}' (kite.enabled={1})" -f $cfg.mode, $cfg.kite.enabled) -ForegroundColor Green

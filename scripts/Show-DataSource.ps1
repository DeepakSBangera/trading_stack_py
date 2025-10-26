param([string]$ConfigPath = 'config\data_source.json')
$ErrorActionPreference = 'Stop'
if (-not (Test-Path $ConfigPath)) { throw "Config not found: $ConfigPath" }
$cfg = Get-Content -Raw -Path $ConfigPath | ConvertFrom-Json
"mode:                  $($cfg.mode)"
"synth.prices_root:     $($cfg.synth.prices_root)"
"synth.fundamentals_root: $($cfg.synth.fundamentals_root)"
"kite.enabled:          $($cfg.kite.enabled)"
"kite.dry_run:          $($cfg.kite.dry_run)"
"kite.cache_root:       $($cfg.kite.cache_root)"
"kite.universe_file:    $($cfg.kite.universe_file)"
"kite.pit_log:          $($cfg.kite.pit_log)"

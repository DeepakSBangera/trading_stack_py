[CmdletBinding()]
param(
  [Parameter(Mandatory=$true)][string] $ModuleKey,
  [string] $Script = '',
  [string] $Args = ''
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$map = @{
  data      = 'modules\DataPipe'
  factors   = 'modules\AlphaFactors'
  signals   = 'modules\AlphaSignals'
  portfolio = 'modules\Portfolio'
  risk      = 'modules\RiskSizing'
  testing   = 'modules\Testing'
  regime    = 'modules\Regime'
  exec      = 'modules\Execution'
  governance= 'modules\Governance'
  monitor   = 'modules\Monitoring'
  dash      = 'modules\Dashboards'
  adapters  = 'modules\Adapters'
}
if(-not $map.ContainsKey($ModuleKey)){ throw "Unknown module: $ModuleKey" }
$root = $map[$ModuleKey]
if($Script){
  $path = Join-Path $root $Script
  if(-not (Test-Path $path)){ throw "Script not found: $path" }
  if($Args){ pwsh $path @Args } else { pwsh $path }
} else {
  Write-Host "Module root: $root"
}

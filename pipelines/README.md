\## W8 → W7 → W9 quick start



You can run tuning (W8), training (W7), and evaluation (W9) end-to-end with either a PowerShell script (Windows-friendly) or a Makefile (cross-platform).



\### Option A — PowerShell



Save as `scripts/w9\_quickstart.ps1`:



```powershell

param(

&nbsp; \[string]$Task = "regression",            # or "classification"

&nbsp; \[string]$TagW8 = "REL\_W8",

&nbsp; \[string]$TagW7 = "REL\_W7",

&nbsp; \[string]$TagW9 = "REL\_W9",

&nbsp; \[string]$W6Dir = ""                      # leave blank to auto-pick newest

)



$ErrorActionPreference = "Stop"



function Get-NewestDir($root) {

&nbsp; if (-not (Test-Path $root)) { throw "Path not found: $root" }

&nbsp; $d = Get-ChildItem $root -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1

&nbsp; if (-not $d) { throw "No subfolders inside $root" }

&nbsp; return $d.FullName

}



\# 1) Resolve W6

if (\[string]::IsNullOrWhiteSpace($W6Dir)) {

&nbsp; $W6Dir = Get-NewestDir "reports\\W6"

}

Write-Host "Using W6:" $W6Dir



\# 2) W8: tuning

python -m trading\_stack\_py.pipelines.tune\_models `

&nbsp; --w6-dir "$W6Dir" `

&nbsp; --task $Task `

&nbsp; --tag $TagW8 `

&nbsp; --outdir reports/W8

$W8 = Get-NewestDir "reports\\W8"; Write-Host "W8:" $W8



\# 3) W7: training

python -m trading\_stack\_py.pipelines.train\_models `

&nbsp; --w6-dir "$W6Dir" `

&nbsp; --task $Task `

&nbsp; --tag $TagW7 `

&nbsp; --outdir reports/W7

$W7 = Get-NewestDir "reports\\W7"; Write-Host "W7:" $W7



\# 4) W9: evaluation/join

python -m trading\_stack\_py.pipelines.evaluate\_models `

&nbsp; --w6-dir "$W6Dir" `

&nbsp; --w7-dir "$W7" `

&nbsp; --tag $TagW9 `

&nbsp; --outdir reports/W9

$W9 = Get-NewestDir "reports\\W9"; Write-Host "W9:" $W9



\# Open summaries (optional)

if (Get-Command code -ErrorAction SilentlyContinue) {

&nbsp; code "$W8\\README.md"

&nbsp; code "$W7\\README.md"

&nbsp; code "$W9\\README.md"

}



powershell -ExecutionPolicy Bypass -File scripts/w9\_quickstart.ps1



powershell -ExecutionPolicy Bypass -File scripts/w9\_quickstart.ps1 `

&nbsp; -Task classification `

&nbsp; -W6Dir "F:\\Projects\\trading\_stack\_py\\reports\\W6\\RELIANCE\_W6\_20250927\_114950" `

&nbsp; -TagW8 RELIANCE\_W8 -TagW7 RELIANCE\_W7 -TagW9 RELIANCE\_W9




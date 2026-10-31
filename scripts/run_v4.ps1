# v4 experiments: matched-epsilon DP with Poisson sampling (improvements 1 and 4),
# real-institution benchmarks (2), wide model + socket link emulation (3).
#   powershell -ExecutionPolicy Bypass -File scripts\run_v4.ps1              # everything
#   powershell -ExecutionPolicy Bypass -File scripts\run_v4.ps1 -Part emu    # one part: emu | dp | real
# The emulation part runs ALONE first, so its measured compute time is not
# inflated by the parallel CPU jobs that follow.
param([ValidateSet("all","emu","dp","real")][string]$Part = "all", [int]$Jobs = 6)
$ErrorActionPreference = "Continue"
$env:PYTHONHASHSEED = "0"
$py = (Resolve-Path ".\.venv\Scripts\python.exe").Path
$seeds = 42..51
New-Item -ItemType Directory -Force results | Out-Null

function Run-Parallel([string[]]$ArgList, [string]$Name) {
  $procs = @(); $i = 0
  foreach ($a in $ArgList) {
    while (($procs | Where-Object { -not $_.HasExited }).Count -ge $Jobs) { Start-Sleep -Seconds 3 }
    $i++
    $procs += Start-Process -FilePath $py -ArgumentList $a -NoNewWindow -PassThru `
      -RedirectStandardOutput "results\v4_${Name}_$i.out" -RedirectStandardError "results\v4_${Name}_$i.log"
  }
  $procs | ForEach-Object { $_.WaitForExit() }
}

if ($Part -in @("all","emu")) {
  Write-Host "[v4] wide model + socket emulation (GPU, alone)"
  & $py -m src.experiments.run_emulation --seeds 42 43 44 --device cuda --out results 2>&1 | Tee-Object -FilePath results\v4_emu.log
}
$cmds = @()
if ($Part -in @("all","dp")) {
  foreach ($s in $seeds) { $cmds += "-m src.experiments.run_dp_matched --seeds $s --device cpu --out results" }
}
if ($Part -in @("all","real")) {
  foreach ($s in $seeds) { $cmds += "-m src.experiments.run_realsites --seeds $s --device cpu --out results" }
}
if ($cmds.Count) {
  # download the open datasets once before the parallel workers start
  if ($Part -in @("all","real")) {
    & $py -c "from src.data.loaders import load_heart, load_eicu_demo; load_heart(); load_eicu_demo()"
  }
  Write-Host "[v4] $($cmds.Count) CPU jobs with $Jobs workers"
  Run-Parallel $cmds $Part
}
& $py -m src.analysis.collect --results results --dataset diabetes --model mlp
Write-Host "V4 DONE ($Part)"

# v5: scale-free TierFed-DP planner, sparsest fixed period, 100-site matched-epsilon DP, eICU demo retry.
#   powershell -ExecutionPolicy Bypass -File scripts\run_v5.ps1
param([int]$Jobs = 6)
$ErrorActionPreference = "Continue"
$env:PYTHONHASHSEED = "0"
$py = (Resolve-Path ".\.venv\Scripts\python.exe").Path
$seeds = 42..51
$L = "results\logs"
# the v4 planner objective is superseded: drop its runs (the v4 table is kept in git history)
Get-ChildItem "$L\*_tierfed_dp_*" -ErrorAction SilentlyContinue | Remove-Item -Force
Get-ChildItem "$L\eicu_demo_*" -ErrorAction SilentlyContinue | Remove-Item -Force

function Run-Parallel([string[]]$ArgList, [string]$Name) {
  $procs = @(); $i = 0
  foreach ($a in $ArgList) {
    while (($procs | Where-Object { -not $_.HasExited }).Count -ge $Jobs) { Start-Sleep -Seconds 3 }
    $i++
    $procs += Start-Process -FilePath $py -ArgumentList $a -NoNewWindow -PassThru `
      -RedirectStandardOutput "results\v5_${Name}_$i.out" -RedirectStandardError "results\v5_${Name}_$i.log"
  }
  $procs | ForEach-Object { $_.WaitForExit() }
}
$cmds = @()
foreach ($s in $seeds) {
  # 20 sites: new planner + K=8 period (other methods already exist and are skipped)
  $cmds += "-m src.experiments.run_dp_matched --seeds $s --clients 20 --device cpu --out results"
  # 100 sites (Dirichlet alpha=1, as in the scaling stage), cohorts of 20 and 50 sites
  $cmds += "-m src.experiments.run_dp_matched --seeds $s --clients 100 --partition dirichlet --alpha 1.0 --grid 0.2:1 0.2:2 0.2:4 0.2:8 0.5:4 --lam-at 0.2:4 --device cpu --out results"
  $cmds += "-m src.experiments.run_realsites --datasets eicu_demo --min-site 40 --seeds $s --device cpu --out results"
}
Write-Host "[v5] $($cmds.Count) CPU jobs with $Jobs workers"
Run-Parallel $cmds all
& $py -m src.analysis.collect --results results --dataset diabetes --model mlp
Write-Host "V5 DONE"

# v6, second batch (CPU): peak-aware staggering with an 8-bit warm-up and the all-rounds quota,
# LAG thresholds that skip, and the timing side channel on the MLP (20 seeds per world).
#   powershell -ExecutionPolicy Bypass -File scripts\run_night2.ps1
# Finished runs are skipped, so the script can be stopped and restarted at any time.
param([int]$Jobs = 6)
$ErrorActionPreference = "Continue"
$env:PYTHONHASHSEED = "0"
$py = (Resolve-Path ".\.venv\Scripts\python.exe").Path
# fetch and cache the Diabetes data once, before parallel jobs start (a fresh clone has no data/)
& $py -c "from src.data import load_dataset; load_dataset('diabetes', n_clients=20, seed=42)"
& $py -m pytest -q tests\test_audit.py
if ($LASTEXITCODE -ne 0) { Write-Host "audit tests failed: not starting"; exit 1 }
$cmds = @()
foreach ($s in 42..51) {
  $cmds += "-m src.experiments.run_audit --stage peak --seeds $s --device cpu --out results"
  $cmds += "-m src.experiments.run_audit --stage frontier2 --seeds $s --device cpu --out results"
  $cmds += "-m src.experiments.run_audit --stage dropout --seeds $s --device cpu --out results"
}
foreach ($s in 42..61) { $cmds += "-m src.experiments.run_audit --stage timing --model mlp --seeds $s --device cpu --out results" }
Write-Host "[v6 night 2] $($cmds.Count) jobs with $Jobs workers"
$procs = @(); $i = 0
foreach ($a in $cmds) {
  while (($procs | Where-Object { -not $_.HasExited }).Count -ge $Jobs) { Start-Sleep -Seconds 3 }
  $i++
  $procs += Start-Process -FilePath $py -ArgumentList $a -NoNewWindow -PassThru `
    -RedirectStandardOutput "results\v6n2_$i.out" -RedirectStandardError "results\v6n2_$i.log"
}
$procs | ForEach-Object { $_.WaitForExit() }
Write-Host "NIGHT2 DONE"

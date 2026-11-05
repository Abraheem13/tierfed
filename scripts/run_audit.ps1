# v6 audit batch: validation split, corrected schedules, frontier, heterogeneity sweep, DP at q = 1.
#   powershell -ExecutionPolicy Bypass -File scripts\run_audit.ps1
# Finished runs are skipped, so the script can be stopped and restarted at any time.
param([int]$Jobs = 6)
$ErrorActionPreference = "Continue"
$env:PYTHONHASHSEED = "0"
$py = (Resolve-Path ".\.venv\Scripts\python.exe").Path
$seeds = 42..51

function Run-Parallel([string[]]$ArgList, [string]$Name) {
  $procs = @(); $i = 0
  foreach ($a in $ArgList) {
    while (($procs | Where-Object { -not $_.HasExited }).Count -ge $Jobs) { Start-Sleep -Seconds 3 }
    $i++
    $procs += Start-Process -FilePath $py -ArgumentList $a -NoNewWindow -PassThru `
      -RedirectStandardOutput "results\v6_${Name}_$i.out" -RedirectStandardError "results\v6_${Name}_$i.log"
  }
  $procs | ForEach-Object { $_.WaitForExit() }
}

# fetch and cache the Diabetes data once, before parallel jobs start (a fresh clone has no data/)
& $py -c "from src.data import load_dataset; load_dataset('diabetes', n_clients=20, seed=42)"
& $py -m pytest -q tests\test_audit.py
if ($LASTEXITCODE -ne 0) { Write-Host "audit tests failed: not starting the batch"; exit 1 }

# fetch DermaMNIST once, so the parallel derma jobs do not download the same file at the same time
& $py -c "import medmnist; [medmnist.DermaMNIST(split=s, download=True) for s in ('train', 'val', 'test')]"

$cmds = @()
# decision-gate experiments first: heterogeneity sweep, then the iso-byte frontier and the main table
foreach ($s in $seeds) { foreach ($a in @("0.1", "0.3", "1", "10")) {
  $cmds += "-m src.experiments.run_audit --stage hetero --alpha $a --seeds $s --device cpu --out results" } }
foreach ($s in $seeds) {
  $cmds += "-m src.experiments.run_audit --stage frontier --seeds $s --device cpu --out results"
  $cmds += "-m src.experiments.run_audit --stage main --seeds $s --device cpu --out results"
}
foreach ($s in $seeds) {
  $cmds += "-m src.experiments.run_audit --stage dp --seeds $s --device cpu --out results"
  $cmds += "-m src.experiments.run_bounds --seeds $s --out results"
  foreach ($a in @("0.1", "0.3", "1", "10")) { $cmds += "-m src.experiments.run_bounds --alpha $a --seeds $s --out results" }
}
foreach ($s in $seeds) { $cmds += "-m src.experiments.run_audit --stage derma --seeds $s --device cpu --out results" }
Write-Host "[v6] $($cmds.Count) jobs with $Jobs workers"
Run-Parallel $cmds all
Write-Host "V6 DONE"

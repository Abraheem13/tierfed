# Full reproduction on one Windows GPU box (no WSL). Resumable: finished runs are skipped.
#   powershell -ExecutionPolicy Bypass -File scripts\reproduce_all.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\reproduce_all.ps1 -Stages 2,3
param(
  [int[]]$Stages = @(1,2,3,4,5,6,7),
  [string]$Device = "cuda",
  [string]$Dataset = "diabetes",
  [string]$Out = "results",
  [int]$Jobs = 4,                    # parallel CPU workers for the tabular MLP arms
  [string]$Seeds = "42 43 44 45 46 47 48 49 50 51"
)
$ErrorActionPreference = "Continue"
$env:PYTHONHASHSEED = "0"
$py = (Resolve-Path ".\.venv\Scripts\python.exe").Path
$seedList = $Seeds -split " "
function Log($m) { Write-Host ""; Write-Host "=========== $(Get-Date -Format HH:mm:ss) $m ===========" }
function Has($s) { $Stages -contains $s }

# Run a list of argument strings as parallel python processes, at most $Jobs at once.
function Run-Parallel([string[]]$ArgList) {
  $procs = @()
  foreach ($a in $ArgList) {
    while (($procs | Where-Object { -not $_.HasExited }).Count -ge $Jobs) { Start-Sleep -Seconds 5 }
    $procs += Start-Process -FilePath $py -ArgumentList $a -NoNewWindow -PassThru
  }
  $procs | ForEach-Object { $_.WaitForExit() }
}

if (Has 1) {
  Log "stage 1: tests + determinism"
  & $py -m pytest tests -q
  if ($LASTEXITCODE -ne 0) { Write-Error "tests failed; stop here" }
  powershell -ExecutionPolicy Bypass -File scripts\check_determinism.ps1 $Dataset cpu
  if ($LASTEXITCODE -ne 0) { Write-Error "determinism check failed; stop here" }
}

if (Has 2) {
  Log "stage 2: main benchmark on $Dataset (9 strategies x 10 seeds, CPU-parallel)"
  $cmds = @()
  foreach ($s in @("fedavg","fedprox","scaffold","fedper","fedlama","adacomm","lag","fixed","tierfed")) {
    $sk = '{}'
    if ($s -eq "tierfed") { $sk = '{\"warmup\":5,\"rho\":0.25}' }
    if ($s -eq "fixed")   { $sk = '{\"k_slow\":5,\"k_med\":2}' }
    foreach ($seed in $seedList) {
      $cmds += "-m src.experiments.run_one --dataset $Dataset --strategy $s --seed $seed --rounds 60 --clients 20 --device cpu --out $Out --strategy-kwargs `"$sk`" --skip-existing"
    }
  }
  Run-Parallel $cmds
  if (Test-Path "data\eicu\patient.csv") {
    Log "stage 2b: eICU with true hospital ids"
    & $py -m src.experiments.run_benchmark --dataset eicu --dataset-kwargs '{\"partition\":\"natural\",\"min_site\":200}' --clients 30 --seeds $seedList --device $Device --out $Out
  } else { Write-Host "eICU not present (data\eicu\patient.csv): skipped; its table will read RUN REQUIRED" }
}

if (Has 3) {
  Log "stage 3: budget / factorial / warmup / compression"
  & $py -m src.experiments.run_ablation --kind budget    --dataset $Dataset --seeds $seedList --device cpu --out $Out
  & $py -m src.experiments.run_ablation --kind factorial --dataset $Dataset --seeds $seedList --device cpu --out $Out
  & $py -m src.experiments.run_ablation --kind warmup    --dataset $Dataset --seeds $seedList --device cpu --out $Out
  & $py -m src.experiments.run_compression               --dataset $Dataset --seeds $seedList --device cpu --out $Out
}

if (Has 4) {
  Log "stage 4: architecture sweep (matched depths) + PneumoniaMNIST"
  & $py -m src.experiments.run_ablation --kind architecture --dataset $Dataset --seeds 42 43 44 45 46 --device $Device --out $Out
  & $py -m src.experiments.run_benchmark --dataset medmnist --model cnn `
      --dataset-kwargs '{\"flag\":\"pneumoniamnist\",\"partition\":\"dirichlet\",\"alpha\":0.3,\"min_size\":50}' `
      --strategies fedavg fedper fedlama fixed tierfed --seeds $seedList --device $Device --out $Out
}

if (Has 5) {
  Log "stage 5: DP utility, inversion attack on real updates, membership inference"
  & $py -m src.experiments.run_privacy --dataset $Dataset --seeds $seedList --device cpu --out $Out
}

if (Has 6) {
  Log "stage 6: scaling 10 -> 100 clients"
  & $py -m src.experiments.run_scaling --dataset $Dataset --client-counts 10 20 50 100 --seeds 42 43 44 --device $Device --out $Out
}

if (Has 7) {
  Log "stage 7: collect -> tables + numbers; figures"
  & $py -m src.analysis.collect --results $Out --dataset $Dataset --model mlp
  & $py -m src.analysis.make_figures --results $Out --dataset $Dataset --model mlp
  & $py scripts\claim_check.py paper\generated\flags.json paper\claims.json
  $missing = Get-ChildItem paper\generated\*.tex | Where-Object { Select-String -Path $_.FullName -Pattern "RUN REQUIRED" -Quiet }
  if ($missing) { Write-Host "tables still waiting for runs:"; $missing | ForEach-Object { Write-Host "  $($_.Name)" } }
  else { Write-Host "all tables generated; build paper\main.tex (e.g. upload the paper folder to Overleaf)" }
}
Write-Host "DONE"

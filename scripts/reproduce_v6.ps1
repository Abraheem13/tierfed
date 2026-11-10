# One-click reproduction of every v6 run and number used in the paper (Windows, CPU).
#   powershell -ExecutionPolicy Bypass -File scripts\reproduce_v6.ps1 -Part gate -Jobs 3
# Parts: core, gate, resnet, centres (four clinical centres), sync (carry-over step), analysis, all.
# Finished runs are skipped, so the script can be stopped and restarted at any time.
param([string]$Part = "all", [int]$Jobs = 3)
$ErrorActionPreference = "Continue"
$env:PYTHONHASHSEED = "0"
$env:SOURCE_DATE_EPOCH = "0"   # figure PDFs without a creation date: byte-identical on every rebuild
Set-Location (Join-Path $PSScriptRoot "..")
$py = (Resolve-Path ".\.venv\Scripts\python.exe").Path
New-Item -ItemType Directory -Force -Path results\v6_logs | Out-Null
$S10 = 42..51; $S5 = 42..46; $S20 = 42..61; $S8 = 42..49
$MAIN = "local fedavg fedprox fedper fedlama adacomm8 lag_uniform lag fixed44 fixed44_st fixed88 tierfed tierfed_st"
# PowerShell variable names are case-insensitive: never reuse $AUD as a loop variable
$AUD = "-m src.experiments.run_audit --device cpu --out results"

function Core {
  $c = @()
  foreach ($s in $S10) { $c += "$AUD --stage main --only $MAIN --seeds $s" }
  foreach ($s in $S10) { $c += "$AUD --stage dp --eps 4 --q 1 --only fedper fixed44 fixed88 oneshot tierfed_dp --seeds $s" }
  foreach ($s in $S10) { $c += "-m src.experiments.run_bounds --seeds $s --out results" }
  foreach ($s in $S5) { $c += "$AUD --stage frontier --only fixed22 fixed66 adacomm4 adacomm16 tierfed_rho0.15 tierfed_rho0.4 --seeds $s" }
  foreach ($s in $S5) { $c += "$AUD --stage frontier2 --only lag_xi5000 lag_xi10000 --seeds $s" }
  foreach ($s in $S5) { $c += "$AUD --stage peak --only tierfed_st_w8 fixed44_st_w8 tierfed_all --seeds $s" }
  return $c
}
function Gate {
  $c = @()
  foreach ($alpha in @("0.1", "0.3", "1", "10")) { foreach ($s in $S5) {
    $c += "$AUD --stage hetero --alpha $alpha --only fedper local fixed44 fixed88 tierfed --seeds $s" } }
  foreach ($s in $S10) { $c += "$AUD --stage timing --model mlp --only tierfed adacomm8 lag_xi2000 lag_xi5000 fixed44 --seeds $s" }
  return $c
}
function Resnet {
  $c = @()
  foreach ($s in $S10) { $c += "$AUD --stage backbone --model resnet --only fedper local fixed44 fixed15 tierfed --seeds $s" }
  foreach ($s in $S8) { $c += "$AUD --stage timing --model resnet --only tierfed --seeds $s" }
  return $c
}
function Centres {
  $c = @()
  foreach ($s in $S20) { $c += "$AUD --stage hospitals --seeds $s" }
  return $c
}
function Sync {
  $c = @()
  foreach ($s in $S10) { $c += "$AUD --stage carryover --only fixed44 fixed88 tierfed tierfed_db adacomm8 --seeds $s" }
  $c += "$AUD --stage carryover --only fedper --seeds 42"
  foreach ($s in $S20) { $c += "$AUD --stage carryover --data heart --only fixed44 fixed88 tierfed tierfed_db adacomm8 --seeds $s" }
  foreach ($s in $S5) { $c += "$AUD --stage carryover --model resnet --only tierfed tierfed_db fixed44 --seeds $s" }
  return $c
}
function Run-Parallel([string[]]$ArgList) {
  Write-Host "[v6 $Part] $($ArgList.Count) jobs with $Jobs workers"
  $procs = @(); $i = 0
  foreach ($job in $ArgList) {
    while (($procs | Where-Object { -not $_.HasExited }).Count -ge $Jobs) { Start-Sleep -Seconds 3 }
    $i++
    $procs += Start-Process -FilePath $py -ArgumentList $job -NoNewWindow -PassThru `
      -RedirectStandardOutput "results\v6_logs\${Part}_$i.out" -RedirectStandardError "results\v6_logs\${Part}_$i.log"
  }
  $procs | ForEach-Object { $_.WaitForExit() }
}
function Analysis {
  & $py -m src.analysis.consistency --results results
  & $py -m src.analysis.crossmachine --results results
  & $py -m src.analysis.audit_report --results results --dataset diabetes | Out-File results\tables\audit_report_stdout.txt
  & $py -m src.analysis.audit_tex --results results --paper paper\generated --dataset diabetes
  & $py -m src.analysis.audit_figures --results results --out paper\figures --dataset diabetes
  & $py -m src.analysis.results_figures --results results --out paper\figures
  & $py -m src.analysis.collect --results results --dataset diabetes --model mlp
  & $py -m src.analysis.make_figures --results results --dataset diabetes --model mlp   # supplement figure
  if (Get-Command pdflatex -ErrorAction SilentlyContinue) {   # MiKTeX or TeX Live; otherwise upload paper\ to Overleaf
    Push-Location paper
    pdflatex -interaction=nonstopmode main.tex | Out-Null; bibtex main | Out-Null
    pdflatex -interaction=nonstopmode main.tex | Out-Null; pdflatex -interaction=nonstopmode main.tex | Out-Null
    pdflatex -interaction=nonstopmode supp.tex | Out-Null; bibtex supp | Out-Null
    pdflatex -interaction=nonstopmode supp.tex | Out-Null; pdflatex -interaction=nonstopmode supp.tex | Out-Null
    Pop-Location
    # the TMLCN manuscript: numbers, tables, figures, both PDFs and the submission archive
    & $py scripts\tmlcn\build.py --analysis --zip
  }
}

# cache the data once, so that parallel jobs never download it at the same time
& $py -c "from src.data import load_dataset; load_dataset('diabetes', n_clients=20, seed=42)" | Out-Null
if ($Part -ne "analysis") {   # repeated runs must be bit-identical before any log is produced
  powershell -ExecutionPolicy Bypass -File scripts\check_determinism.ps1 diabetes cpu
  if ($LASTEXITCODE -ne 0) { throw "determinism check failed: fix the environment first" }
}
switch ($Part) {
  "core"     { Run-Parallel (Core) }
  "gate"     { Run-Parallel (Gate) }
  "resnet"   { Run-Parallel (Resnet) }
  "centres"  { Run-Parallel (Centres) }
  "sync"     { Run-Parallel (Sync) }
  "analysis" { Analysis }
  "all"      { Run-Parallel ((Core) + (Gate) + (Resnet) + (Centres) + (Sync)); Analysis }
}
Write-Host "V6 $Part DONE"

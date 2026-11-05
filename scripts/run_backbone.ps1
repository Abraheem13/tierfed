# v6 backbone stage (GPU): ResNet and FT-Transformer with the validation split, 10 seeds.
#   powershell -ExecutionPolicy Bypass -File scripts\run_backbone.ps1
# Finished runs are skipped, so the script can be stopped and restarted at any time.
param([int]$Jobs = 3, [string]$Device = "cuda")
$ErrorActionPreference = "Continue"
$env:PYTHONHASHSEED = "0"
$py = (Resolve-Path ".\.venv\Scripts\python.exe").Path
# fetch and cache the Diabetes data once, before parallel jobs start (a fresh clone has no data/)
& $py -c "from src.data import load_dataset; load_dataset('diabetes', n_clients=20, seed=42)"
$procs = @(); $i = 0
foreach ($m in @("resnet", "transformer")) { foreach ($s in 42..51) {
  while (($procs | Where-Object { -not $_.HasExited }).Count -ge $Jobs) { Start-Sleep -Seconds 3 }
  $i++
  $procs += Start-Process -FilePath $py -NoNewWindow -PassThru `
    -ArgumentList "-m src.experiments.run_audit --stage backbone --model $m --seeds $s --device $Device --out results" `
    -RedirectStandardOutput "results\v6_bb_$i.out" -RedirectStandardError "results\v6_bb_$i.log"
} }
# timing side channel where TierFed differentiates: ResNet, shifted vs clean target site, 20 seeds
foreach ($s in 42..61) {
  while (($procs | Where-Object { -not $_.HasExited }).Count -ge $Jobs) { Start-Sleep -Seconds 3 }
  $i++
  $procs += Start-Process -FilePath $py -NoNewWindow -PassThru `
    -ArgumentList "-m src.experiments.run_audit --stage timing --model resnet --seeds $s --device $Device --out results" `
    -RedirectStandardOutput "results\v6_bb_$i.out" -RedirectStandardError "results\v6_bb_$i.log"
}
$procs | ForEach-Object { $_.WaitForExit() }
Write-Host "BACKBONE DONE"

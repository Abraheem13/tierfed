# Runs one configuration twice and asserts bit-identical summaries.
#   powershell -ExecutionPolicy Bypass -File scripts\check_determinism.ps1 [dataset] [device]
param([string]$Dataset = "diabetes", [string]$Device = "cpu")
$ErrorActionPreference = "Stop"
$env:PYTHONHASHSEED = "0"
$py = ".\.venv\Scripts\python.exe"
$out = "results\determinism"
if (Test-Path $out) { Remove-Item -Recurse -Force $out }
foreach ($rep in @("a", "b")) {
  & $py -m src.experiments.run_one --dataset $Dataset --strategy tierfed --seed 42 --rounds 8 --clients 20 `
      --device $Device --out "$out\$rep" --strategy-kwargs '{\"warmup\":2,\"rho\":0.25}' | Out-Null
}
$code = @"
import json, glob, sys
a = json.load(open(glob.glob('results/determinism/a/logs/*_summary.json')[0]))['summary']
b = json.load(open(glob.glob('results/determinism/b/logs/*_summary.json')[0]))['summary']
keys = ['fed_auroc_final', 'fed_auprc_final', 'total_upload_mb', 'divergence_total_mean']
diff = {k: abs(a[k] - b[k]) for k in keys}
print('determinism check:', diff)
if max(diff.values()) > 0:
    print('FAIL: repeated runs differ; fix the environment before producing logs'); sys.exit(1)
print('PASS: repeated runs are bit-identical')
"@
$code | & $py -
exit $LASTEXITCODE

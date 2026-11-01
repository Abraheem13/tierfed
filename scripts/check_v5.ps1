$L = "results\logs"
$d20  = (Get-ChildItem "$L\*_C20_*_dpe*_summary.json" -ErrorAction SilentlyContinue).Count
$d100 = (Get-ChildItem "$L\*_C100_*_dpe*_summary.json" -ErrorAction SilentlyContinue).Count
$eicu = (Get-ChildItem "$L\eicu_demo_*_real_summary.json" -ErrorAction SilentlyContinue).Count
Write-Host "dp 20 sites $d20/280 | dp 100 sites $d100/280 | eicu demo $eicu/90"
Get-ChildItem results\v5_*.log -ErrorAction SilentlyContinue | Select-String -Pattern "Error" | Select-Object -First 5

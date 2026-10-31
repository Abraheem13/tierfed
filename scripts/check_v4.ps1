# progress of the v4 runs
$L = "results\logs"
$dp   = (Get-ChildItem "$L\*_dpe*_summary.json" -ErrorAction SilentlyContinue).Count
$real = (Get-ChildItem "$L\*_real_summary.json" -ErrorAction SilentlyContinue).Count
$wide = (Get-ChildItem "$L\*_wide_summary.json" -ErrorAction SilentlyContinue).Count
$emu  = (Get-ChildItem "results\emulation\*.json" -ErrorAction SilentlyContinue).Count
Write-Host "dp-matched $dp/220 | real-sites $real/180 | wide $wide/15 | emulation $emu/35"
Get-ChildItem results\v4_*.log -ErrorAction SilentlyContinue | Select-String -Pattern "Traceback|Error" | Select-Object -First 5

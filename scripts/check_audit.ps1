# progress of the v6 audit batch
$L = "results\logs"
function N($p) { (Get-ChildItem "$L\$p" -ErrorAction SilentlyContinue).Count }
$hb   = N "*_au_het_*_bound_summary.json"
$het  = (N "*_au_het_*_summary.json") - $hb
$dp   = N "*_au_dpe*_summary.json"
$der  = N "*_au_derma_*_summary.json"
$bnd  = N "*_au*_bound_summary.json"
$all  = N "*_au_*_summary.json"
$bb   = N "*_au_bb_*_summary.json"
$pk   = (N "*_au_*_w8_summary.json") + (N "*_au_tierfed_all*_summary.json") - (N "*_au_tierfed_all_w8_summary.json")
$main = $all - $het - $dp - $der - $bnd - $bb - $pk
$tm   = N "*_au_tm_*_summary.json"
$f2   = (N "*_au_lag_xi2000_summary.json") + (N "*_au_lag_xi5000_summary.json") + (N "*_au_lag_xi10000_summary.json")
$drp  = N "*_au_drop*_summary.json"
$main = $main - $tm - $f2 - $drp
Write-Host "night 1: hetero $het/520 | main+frontier $main/490 | dp $dp/380 | bounds $bnd/100 | derma $der/50"
Write-Host "night 2: backbone $bb/260 | peak $pk/90 | lag $f2/30 | dropout $drp/80 | timing $tm/200"
Get-ChildItem results\v6_*.log -ErrorAction SilentlyContinue | Select-String -Pattern "Traceback|Error" | Select-Object -First 5

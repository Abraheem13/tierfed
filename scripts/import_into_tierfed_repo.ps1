# PowerShell equivalent of import_into_tierfed_repo.sh.
# Run from inside the tierfed folder:   powershell -ExecutionPolicy Bypass -File scripts\import_into_tierfed_repo.ps1
# Moves this tree into https://github.com/Abraheem13/tierfed (history kept) as a commit by the configured git user.
$ErrorActionPreference = "Stop"
$Src    = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Dst    = if ($env:DST)    { $env:DST }    else { Join-Path (Split-Path $Src -Parent) "tierfed-import" }
$Remote = if ($env:REMOTE) { $env:REMOTE } else { "https://github.com/Abraheem13/tierfed.git" }
$Branch = if ($env:BRANCH) { $env:BRANCH } else { "main" }

if (-not (git config user.name) -or -not (git config user.email)) {
  Write-Error "Set your identity first: git config --global user.name 'Abraheem Rashid'; git config --global user.email '<your email>'"
}

if (Test-Path $Dst) { Remove-Item -Recurse -Force $Dst }
git clone $Remote $Dst
Set-Location $Dst
git checkout $Branch

# replace the working tree, keep .git
Get-ChildItem -Force | Where-Object { $_.Name -ne ".git" } | Remove-Item -Recurse -Force

# copy v3 in, skipping local scratch
$exclude = @(".venv", "__pycache__", ".pytest_cache", ".git")
Get-ChildItem -Path $Src -Force | Where-Object { $exclude -notcontains $_.Name } | ForEach-Object {
  Copy-Item -Path $_.FullName -Destination (Join-Path $Dst $_.Name) -Recurse -Force
}
foreach ($d in @("results\logs", "results\tables", "results\figures")) {
  New-Item -ItemType Directory -Force -Path (Join-Path $Dst $d) | Out-Null
  New-Item -ItemType File -Force -Path (Join-Path $Dst "$d\.gitkeep") | Out-Null
}
Get-ChildItem -Path (Join-Path $Dst "results\logs") -Exclude ".gitkeep" | Remove-Item -Recurse -Force
Get-ChildItem -Path (Join-Path $Dst "results\tables") -Exclude ".gitkeep" | Remove-Item -Recurse -Force
Get-ChildItem -Path (Join-Path $Dst "paper") -Include "*.pdf","*.aux","*.log","*.bbl","*.blg" -Recurse | Remove-Item -Force -ErrorAction SilentlyContinue

git add -A
git commit -m "TierFed v3: audited rewrite" -m "Corrects the defects of v2 (DP not applied to three baselines and invalid accounting; SCAFFOLD stub; unsupported scaling table; mismatched architecture depths; undisclosed imaging configuration; mislabelled inversion attack; nondeterministic runs; no patient dedup) and adds the network cost model, attack-surface evaluation, exact budget solver, AdaComm/LAG baselines, and a manuscript whose every number is generated from per-seed logs. All v2 result logs removed."
git push origin $Branch
Write-Host "done: $(git rev-parse --short HEAD) pushed to $Remote $Branch"

# Brings the v13 state of this project into your clone of https://github.com/Abraheem13/tierfed
# as one commit by your own git identity. Run it from the ROOT of that clone:
#   powershell -ExecutionPolicy Bypass -File scripts\import_into_tierfed_repo.ps1
# What it does (nothing else is touched):
#   1. fetches the tierfed/ folder of the source branch and unpacks it over the clone
#   2. before that, removes results\logs\*_au_* (the v6 logs): the Linux runs behind the paper
#      replace them, and the laptop's runs live on in results\crossmachine\laptop_win11\logs
#   3. earlier-protocol logs, your other files and the git history stay as they are
#   4. commits and pushes
param(
  [string]$Source = "https://github.com/abraheemrashid99-sys/temp.git",
  [string]$Branch = "claude/brave-fermi-g44qr8",
  [switch]$NoPush
)
$ErrorActionPreference = "Stop"
if (-not (Test-Path ".git") -or -not (Test-Path "src") -or -not (Test-Path "paper")) {
  throw "Run this from the root of your tierfed clone (the folder with .git, src and paper)."
}
if (-not (git config user.name) -or -not (git config user.email)) {
  throw "Set your identity first: git config --global user.name 'Abraheem Rashid'; git config --global user.email '<your email>'"
}
git pull
git fetch $Source $Branch
$tar = Join-Path $env:TEMP "tierfed_v13.tar"
git archive -o $tar FETCH_HEAD tierfed
Get-ChildItem results\logs -Filter "*_au_*" -ErrorAction SilentlyContinue | Remove-Item -Force
tar -xf $tar --strip-components=1
Remove-Item $tar
git add -A
git commit -m "v13: final v6 results, cross-machine check, paper and supplement"
if (-not $NoPush) { git push }
Write-Host "done: $(git rev-parse --short HEAD)"

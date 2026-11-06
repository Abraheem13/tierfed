#!/usr/bin/env bash
# Brings the v13 state of this project into your clone of https://github.com/Abraheem13/tierfed
# as one commit by your own git identity. Run it from the ROOT of that clone:
#   bash scripts/import_into_tierfed_repo.sh [--no-push]
# What it does (nothing else is touched):
#   1. fetches the tierfed/ folder of the source branch and unpacks it over the clone
#   2. before that, removes results/logs/*_au_* (the v6 logs): the Linux runs behind the paper
#      replace them, and the laptop's runs live on in results/crossmachine/laptop_win11/logs
#   3. earlier-protocol logs, your other files and the git history stay as they are
#   4. commits and pushes
set -euo pipefail
SOURCE="${SOURCE:-https://github.com/abraheemrashid99-sys/temp.git}"
BRANCH="${BRANCH:-claude/brave-fermi-g44qr8}"
[ -d .git ] && [ -d src ] && [ -d paper ] || { echo "run this from the root of your tierfed clone"; exit 1; }
git config user.name >/dev/null && git config user.email >/dev/null || {
  echo "set your identity first: git config --global user.name 'Abraheem Rashid'; git config --global user.email '<your email>'"; exit 1; }
git pull
git fetch "$SOURCE" "$BRANCH"
tmp="$(mktemp)"
git archive -o "$tmp" FETCH_HEAD tierfed
find results/logs -maxdepth 1 -name '*_au_*' -delete 2>/dev/null || true
tar -xf "$tmp" --strip-components=1
rm -f "$tmp"
git add -A
git commit -m "Import the latest paper, code and run logs"
[ "${1:-}" = "--no-push" ] || git push
echo "done: $(git rev-parse --short HEAD)"

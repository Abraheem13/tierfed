#!/usr/bin/env bash
# Run this ON YOUR OWN MACHINE to move TierFed v3 into github.com/Abraheem13/tierfed
# as a commit authored by you. Nothing in this script contacts any service other
# than GitHub, and the commit carries only your git identity.
#
#   bash scripts/import_into_tierfed_repo.sh            # from the temp repo checkout
#
# What it does:
#   1. clones Abraheem13/tierfed (default branch) into ../tierfed-import
#   2. removes everything in it except .git
#   3. copies this tierfed/ tree in
#   4. commits as the configured git user and pushes
set -euo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)"
DST="${DST:-$SRC/../tierfed-import}"
REMOTE="${REMOTE:-git@github.com:Abraheem13/tierfed.git}"
BRANCH="${BRANCH:-main}"

if ! git config user.name >/dev/null || ! git config user.email >/dev/null; then
  echo "set your identity first: git config --global user.name 'Abraheem Rashid'; git config --global user.email '<your email>'"; exit 1
fi

rm -rf "$DST"
git clone "$REMOTE" "$DST"
cd "$DST"
git checkout "$BRANCH"
# keep history; replace the working tree
find . -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
# copy v3 (excluding local scratch)
(cd "$SRC" && tar --exclude=.venv --exclude=__pycache__ --exclude=.pytest_cache --exclude='results/logs/*' \
     --exclude='results/tables/*' --exclude='paper/*.pdf' --exclude='paper/*.aux' -cf - .) | tar -xf -
mkdir -p results/logs results/tables results/figures
touch results/logs/.gitkeep results/tables/.gitkeep results/figures/.gitkeep
git add -A
git commit -m "TierFed v3: audited rewrite

Corrects the defects of v2 (DP not applied to three baselines and invalid
accounting; SCAFFOLD stub; unsupported scaling table; mismatched
architecture depths; undisclosed imaging configuration; mislabelled
inversion attack; nondeterministic runs; no patient dedup) and adds the
network cost model, attack-surface evaluation, exact budget solver,
AdaComm/LAG baselines, and a manuscript whose every number is generated
from per-seed logs. All v2 result logs removed."
git push origin "$BRANCH"
echo "done: $(git rev-parse --short HEAD) pushed to $REMOTE $BRANCH"

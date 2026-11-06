#!/usr/bin/env bash
# Experiments that close the two remaining evidence gaps of the TMLCN manuscript.
# Run from the repository root with the project's environment active and
# data/raw/diabetic_data.csv in place. Every run writes its logs to results/logs
# under the same naming scheme as the existing runs, so the analysis scripts pick
# them up unchanged.
set -euo pipefail

# 1. ResNet backbone to ten seeds (seeds 42-47 exist): the only setting where Psi is
#    large. 5 configurations x 4 seeds = 20 ResNet runs.
python -m src.experiments.run_audit --stage backbone --model resnet \
  --seeds 48 49 50 51 --only fedper local fixed44 fixed15 tierfed

# 2. Timing side channel where TierFed adapts (ResNet), 20 runs per world:
#    TierFed and the data-independent control Fixed (4,4), clean and shifted worlds.
#    2 configurations x 2 worlds x 20 seeds = 80 ResNet runs.
python -m src.experiments.run_audit --stage timing --model resnet \
  --seeds $(seq 42 61) --only tierfed fixed44

# 3. Timing side channel on the MLP to 20 runs per world (seeds 42-46 exist):
#    5 configurations x 2 worlds x 15 seeds = 150 MLP runs.
python -m src.experiments.run_audit --stage timing --model mlp --seeds $(seq 47 61)

# 4. Regenerate the tables and numbers from all logs, then the TMLCN analyses.
python -m src.analysis.audit_tex
python -I scripts/tmlcn/selection_table.py results/logs > scripts/tmlcn/selection_table.out
python -I scripts/tmlcn/dp_local_paired.py results/logs > scripts/tmlcn/dp_local_paired.out
python -I scripts/tmlcn/check_numbers.py paper/tmlcn/main.tex

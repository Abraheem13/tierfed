#!/usr/bin/env bash
# Full reproduction on one GPU box (tested target: RTX 5080, 16 GB).
# Every run is resumable: existing *_summary.json files are skipped.
#
#   bash scripts/reproduce_all.sh            # all stages
#   STAGES="1 2" bash scripts/reproduce_all.sh
#
# Approximate cost (60 rounds, 20 clients): MLP run ~3 min CPU / ~2 min GPU,
# TabResNet ~4 min, FT-Transformer ~20 min GPU, CNN ~4 min GPU.
set -euo pipefail
export PYTHONHASHSEED=0
cd "$(dirname "$0")/.."
DEV="${DEVICE:-cuda}"
DS="${DATASET:-diabetes}"
OUT="${OUT:-results}"
SEEDS="${SEEDS:-42 43 44 45 46 47 48 49 50 51}"
STAGES="${STAGES:-1 2 3 4 5 6 7}"
J="${CPU_JOBS:-4}"          # parallel CPU workers for the tabular MLP arms
log(){ echo; echo "=========== $(date +%H:%M:%S) $1 ==========="; }
has(){ [[ " $STAGES " == *" $1 "* ]]; }

# --- stage 1: environment + determinism ------------------------------------
if has 1; then
  log "stage 1: tests + determinism"
  python -m pytest tests -q
  bash scripts/check_determinism.sh "$DS" "$DEV"
fi

# --- stage 2: main benchmark (tabular) ---------------------------------------
# Tabular MLP runs are CPU-bound; run them in parallel on CPU cores and keep
# the GPU for the deep arms.
if has 2; then
  log "stage 2: main benchmark on $DS (9 strategies x 10 seeds)"
  CMDS=()
  for s in fedavg fedprox scaffold fedper fedlama adacomm lag fixed tierfed; do
    SK='{}'; [ "$s" = "tierfed" ] && SK='{"warmup":5,"rho":0.25}'; [ "$s" = "fixed" ] && SK='{"k_slow":5,"k_med":2}'
    for seed in $SEEDS; do
      CMDS+=("python -m src.experiments.run_one --dataset $DS --strategy $s --seed $seed --rounds 60 --clients 20 --device cpu --out $OUT --strategy-kwargs '$SK' --skip-existing")
    done
  done
  printf '%s\n' "${CMDS[@]}" | xargs -P "$J" -I {} bash -c '{}'
  if [ -f data/eicu/patient.csv ]; then
    log "stage 2b: eICU with true hospital ids"
    python -m src.experiments.run_benchmark --dataset eicu --dataset-kwargs '{"partition":"natural","min_site":200}' \
      --clients 30 --seeds $SEEDS --device "$DEV" --out "$OUT"
  else
    echo "eICU not present (data/eicu/patient.csv): skipping; the paper's eICU table will read RUN REQUIRED"
  fi
fi

# --- stage 3: budget sweep, factorial, warmup, compression --------------------
if has 3; then
  log "stage 3: budget / factorial / warmup / compression"
  python -m src.experiments.run_ablation --kind budget    --dataset "$DS" --seeds $SEEDS --device cpu --out "$OUT"
  python -m src.experiments.run_ablation --kind baselines --dataset "$DS" --seeds $SEEDS --device cpu --out "$OUT"
  python -m src.experiments.run_ablation --kind factorial --dataset "$DS" --seeds $SEEDS --device cpu --out "$OUT"
  python -m src.experiments.run_ablation --kind warmup    --dataset "$DS" --seeds $SEEDS --device cpu --out "$OUT"
  python -m src.experiments.run_compression               --dataset "$DS" --seeds $SEEDS --device cpu --out "$OUT"
fi

# --- stage 4: architectures (GPU) + imaging (GPU) -----------------------------
if has 4; then
  log "stage 4: architecture sweep (matched depths) + PneumoniaMNIST"
  python -m src.experiments.run_ablation --kind architecture --dataset "$DS" --seeds 42 43 44 45 46 --device "$DEV" --out "$OUT"
  python -m src.experiments.run_benchmark --dataset medmnist --model cnn \
    --dataset-kwargs '{"flag":"pneumoniamnist","partition":"dirichlet","alpha":0.3,"min_size":50}' \
    --strategies fedavg fedper fedlama fixed tierfed --seeds $SEEDS --device "$DEV" --out "$OUT"
fi

# --- stage 5: privacy + attack surface ----------------------------------------
if has 5; then
  log "stage 5: DP utility, inversion attack on real updates, membership inference"
  python -m src.experiments.run_privacy --dataset "$DS" --seeds $SEEDS --device cpu --out "$OUT"
fi

# --- stage 6: scaling ---------------------------------------------------------
if has 6; then
  log "stage 6: scaling 10 -> 100 clients"
  python -m src.experiments.run_scaling --dataset "$DS" --client-counts 10 20 50 100 --seeds 42 43 44 --device "$DEV" --out "$OUT"
fi

# --- stage 7: tables, figures, paper ------------------------------------------
if has 7; then
  log "stage 7: collect -> tables + numbers; figures; paper"
  python -m src.analysis.collect --results "$OUT" --dataset "$DS" --model mlp
  python -m src.analysis.make_figures --results "$OUT" --dataset "$DS" --model mlp
  python scripts/claim_check.py paper/generated/flags.json paper/claims.json || true
  (cd paper && make) || echo "LaTeX build failed or not installed; generated tables are in paper/generated"
fi
echo DONE

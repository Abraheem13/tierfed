#!/usr/bin/env bash
# Parallel launcher. Tabular runs are overhead-bound, so N parallel processes
# scale close to linearly with cores.
#
#   bash scripts/run_parallel.sh <experiment> [jobs] [dataset] [device]
#
# experiment: benchmark | factorial | periods | loss | compression | privacy
set -euo pipefail
EXP="${1:-benchmark}"
JOBS="${2:-8}"
DS="${3:-diabetes}"
DEV="${4:-cpu}"
ROUNDS="${ROUNDS:-60}"
CLIENTS="${CLIENTS:-20}"
SEEDS="${SEEDS:-42 43 44 45 46 47 48 49 50 51}"
OUT="${OUT:-results}"

run() { printf '%s\n' "$@"; }
CMDS=()

case "$EXP" in
  benchmark)
    for s in fedavg fedprox fedper fedlama scaffold nested; do
      for seed in $SEEDS; do
        CMDS+=("python -m src.experiments.run_one --dataset $DS --strategy $s --seed $seed \
--rounds $ROUNDS --clients $CLIENTS --device $DEV --out $OUT --threads 1")
      done
    done ;;
  factorial)
    for seed in $SEEDS; do
      CMDS+=("python -m src.experiments.run_one --dataset $DS --strategy fedavg --seed $seed --rounds $ROUNDS --clients $CLIENTS --device $DEV --out $OUT --tag _fac_head0_sched0 --threads 1")
      CMDS+=("python -m src.experiments.run_one --dataset $DS --strategy fedper --seed $seed --rounds $ROUNDS --clients $CLIENTS --device $DEV --out $OUT --tag _fac_head1_sched0 --threads 1")
      CMDS+=("python -m src.experiments.run_one --dataset $DS --strategy nested --seed $seed --rounds $ROUNDS --clients $CLIENTS --device $DEV --out $OUT --tag _fac_head0_sched1 --strategy-kwargs '{\"private_head\":false,\"schedule\":true}' --threads 1")
      CMDS+=("python -m src.experiments.run_one --dataset $DS --strategy nested --seed $seed --rounds $ROUNDS --clients $CLIENTS --device $DEV --out $OUT --tag _fac_head1_sched1 --strategy-kwargs '{\"private_head\":true,\"schedule\":true}' --threads 1")
    done ;;
  periods)
    for ks in 2 5 10; do for km in 1 2 4 8; do for seed in $SEEDS; do
      CMDS+=("python -m src.experiments.run_one --dataset $DS --strategy nested --seed $seed --rounds $ROUNDS --clients $CLIENTS --device $DEV --out $OUT --tag _ks${ks}km${km} --strategy-kwargs '{\"k_slow\":$ks,\"k_med\":$km}' --threads 1")
    done; done; done ;;
  loss)
    for l in weighted_ce focal ldam cb; do for s in fedavg nested; do for seed in $SEEDS; do
      CMDS+=("python -m src.experiments.run_one --dataset $DS --strategy $s --loss $l --seed $seed --rounds $ROUNDS --clients $CLIENTS --device $DEV --out $OUT --tag _loss_$l --threads 1")
    done; done; done ;;
  compression)
    for c in none quant8 quant4 topk stc sketch; do for s in fedavg nested; do for seed in 42 43 44 45 46; do
      CMDS+=("python -m src.experiments.run_one --dataset $DS --strategy $s --compressor $c --seed $seed --rounds $ROUNDS --clients $CLIENTS --device $DEV --out $OUT --tag _comp_$c --threads 1")
    done; done; done ;;
  privacy)
    for sg in 0 0.5 1.0 2.0; do for s in fedavg nested; do for seed in 42 43 44 45 46; do
      CMDS+=("python -m src.experiments.run_one --dataset $DS --strategy $s --dp-sigma $sg --seed $seed --rounds $ROUNDS --clients $CLIENTS --device $DEV --out $OUT --tag _dp$sg --threads 1")
    done; done; done ;;
  *) echo "unknown experiment '$EXP'"; exit 1 ;;
esac

echo "launching ${#CMDS[@]} runs with $JOBS parallel workers ($EXP / $DS)"
printf '%s\n' "${CMDS[@]}" | xargs -P "$JOBS" -I {} bash -c '{}'
echo "done: $EXP"

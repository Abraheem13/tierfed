#!/usr/bin/env bash
set -uo pipefail
J="${1:-6}"; DS="${2:-diabetes}"; OUT="${OUT:-results}"
SEEDS="${SEEDS:-42 43 44 45 46 47 48 49 50 51}"
run_job() {
  python -m src.experiments.run_one --dataset "$2" --strategy "$3" --seed "$4" \
    --rounds 60 --clients 20 --device cpu --out "$5" --tag "$6" \
    --strategy-kwargs "$7" --threads 1
}
export -f run_job
N=0
for rho in 0.6 0.4 0.25; do
  for s in $SEEDS; do
    echo "run_job x $DS tierfed $s $OUT _tf_rho${rho} {\"warmup\":5,\"rho\":${rho}}"
    N=$((N+1))
  done
done
for s in $SEEDS; do
  echo "run_job x $DS nested $s $OUT _tf_warm {\"k_slow\":5,\"k_med\":2,\"warmup\":8}"
  N=$((N+1))
done
echo "launching $N runs with $J workers" >&2

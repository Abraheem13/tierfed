#!/usr/bin/env bash
set -uo pipefail
echo "=== compression remainder (CPU) ==="
bash scripts/run_parallel.sh compression 4 diabetes cpu

echo "=== arch generalisation: TierFed on resnet/transformer (GPU) ==="
for m in resnet transformer; do
  for s in 42 43 44; do
    python -m src.experiments.run_one --dataset diabetes --model "$m" --strategy tierfed \
      --seed "$s" --rounds 60 --clients 20 --device cuda --out results \
      --tag _arch_tf --strategy-kwargs '{"warmup":5,"rho":0.25}' --threads 2
  done
done

echo "=== imaging: dermamnist, alpha=0.1 (GPU) ==="
for s in 42 43 44 45 46; do
  for st in fedavg fedper tierfed; do
    SK='{}'; [ "$st" = "tierfed" ] && SK='{"warmup":5,"rho":0.25}'
    python -m src.experiments.run_one --dataset medmnist --model cnn --strategy "$st" \
      --seed "$s" --rounds 60 --clients 20 --device cuda --out results --tag _img2 \
      --dataset-kwargs '{"flag":"dermamnist","partition":"dirichlet","alpha":0.1}' \
      --strategy-kwargs "$SK" --threads 2
  done
done
echo ALL_DONE

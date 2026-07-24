#!/usr/bin/env bash
set -uo pipefail
J="${CPU_JOBS:-6}"; DS="${DATASET:-diabetes}"; OUT="${OUT:-results}"
sec(){ echo; echo "=========== $(date +%H:%M:%S) $1 ==========="; }

sec "factorial"
bash scripts/run_parallel.sh factorial "$J" "$DS" cpu
python -m src.analysis.collect --results "$OUT" --dataset "$DS" --model mlp

sec "stage2"
bash scripts/run_parallel.sh periods     "$J" "$DS" cpu
bash scripts/run_parallel.sh loss        "$J" "$DS" cpu
bash scripts/run_parallel.sh compression "$J" "$DS" cpu
bash scripts/run_parallel.sh privacy     "$J" "$DS" cpu
python -m src.analysis.collect --results "$OUT" --dataset "$DS" --model mlp
python -m src.experiments.run_scaling --dataset "$DS" --client-counts 10 20 50 100 --seeds 42 43 44 --out "$OUT" || true
python -m src.experiments.run_privacy --dataset "$DS" --seeds 42 43 44 45 46 --out "$OUT" || true

GPU=$(python -c "import torch;print(int(torch.cuda.is_available()))" 2>/dev/null || echo 0)
if [ "$GPU" = "1" ]; then
  sec "stage3 GPU"
  python -m src.experiments.run_ablation --kind architecture --dataset "$DS" --device cuda --seeds 42 43 44 --out "$OUT" || true
  python -m src.experiments.run_benchmark --dataset medmnist --model cnn --strategies fedavg fedper nested --seeds 42 43 44 45 46 --device cuda --out "$OUT" || true
else
  echo "SKIPPING STAGE 3 — no GPU visible"
fi

sec "figures + commit"
python -m src.analysis.collect --results "$OUT" || true
python -m src.analysis.make_figures --results "$OUT" --dataset "$DS" --model mlp || true
sed -i '/^results\/logs/d;/gitkeep/d' .gitignore 2>/dev/null || true
git add -A && git commit -m "Full experiment run $(date -u +%Y-%m-%dT%H:%MZ)" || echo "nothing to commit"
git push origin main || echo "PUSH FAILED — commit is local, push manually"
echo "DONE"

# Runbook — environment, commands, and compute budget

## 0. Hardware reality check (read this first)

The tabular experiments are **not GPU-bound**. A 36,930-parameter MLP with
batch size 64 spends its time in the Python loop and the dataloader, not in
matrix multiplication. Measured at diabetes scale (≈79k rows, 20 clients):

| Model | s/round | Where it runs best |
|---|---|---|
| MLP (36.9k params) | ~3.2 | CPU; GPU gives ≲1.5× |
| TabResNet (277k) | ~3.8 | CPU or GPU |
| FT-Transformer | ~60–200 | **GPU essential** (~62× the MLP on CPU) |
| SmallCNN (MedMNIST) | — | **GPU essential** |

**Therefore: parallel CPU processes buy you far more than a bigger GPU.**
Run 8–16 jobs concurrently (`scripts/run_parallel.sh`); reserve the GPU for the
FT-Transformer and MedMNIST arms.

---

## 1. Environment

```bash
# clone / unzip, then:
cd nested-fl-healthcare
bash scripts/setup_env.sh venv     # or: bash scripts/setup_env.sh conda
source .venv/bin/activate          # conda: conda activate nfl
python -m pytest tests/ -q         # expect: 31 passed
```

Manual equivalent:

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip wheel
pip install torch --index-url https://download.pytorch.org/whl/cu121   # cu118 for older drivers
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 2. Data

```bash
export NFL_DATA_ROOT=$PWD/data

# diabetes — automatic (UCI id 296)
python -c "from src.data import load_dataset; load_dataset('diabetes', n_clients=20)"

# MedMNIST — automatic on first use
python -c "from src.data import load_dataset; load_dataset('medmnist', n_clients=20)"

# eICU / MIMIC — credentialed PhysioNet access required; download yourself:
#   data/eicu/patient.csv
#   data/mimic/admissions.csv
```

## 3. Staged execution

### Stage 1 — the decisive runs (do these first, ~2–4 h on 8 workers)

These determine what the paper can claim. Nothing else matters until they land.

```bash
bash scripts/run_parallel.sh benchmark 8 diabetes cpu     # 60 runs: all 6 strategies x 10 seeds
bash scripts/run_parallel.sh factorial 8 diabetes cpu     # 40 runs: the 2x2 attribution
python -m src.experiments.run_benchmark --dataset diabetes --seeds 42 43 44 45 46 47 48 49 50 51
```

Then read `results/tables/benchmark_diabetes_mlp_tests.csv` — specifically the
`nested_vs_fedper` rows and the `significant_holm` column.

### Stage 2 — supporting evidence (~6–10 h on 8–16 workers)

```bash
bash scripts/run_parallel.sh periods     12 diabetes cpu
bash scripts/run_parallel.sh loss        12 diabetes cpu
bash scripts/run_parallel.sh compression 12 diabetes cpu
bash scripts/run_parallel.sh privacy     12 diabetes cpu
python -m src.experiments.run_scaling --dataset diabetes --client-counts 10 20 50 100
python -m src.experiments.run_privacy --dataset diabetes --skip-inversion=false
```

### Stage 3 — external validity (the objection-killers)

```bash
# true hospital IDs — removes the pseudo-hospital criticism outright
python -m src.experiments.run_benchmark --dataset eicu --partition natural \
       --clients 40 --seeds 42 43 44 45 46 47 48 49 50 51 --device cuda

# imaging modality + deep architecture (GPU)
python -m src.experiments.run_benchmark --dataset medmnist --model cnn \
       --clients 20 --seeds 42 43 44 45 46 47 48 49 50 51 --device cuda

# architecture generalisation (GPU for the transformer arm)
python -m src.experiments.run_ablation --kind architecture --device cuda --seeds 42 43 44 45 46
```

### Stage 4 — figures and tables

```bash
python -m src.analysis.make_figures --results results --dataset diabetes --model mlp
```

---

## 4. Compute budget

Per-run cost at diabetes scale, 60 rounds, single process:
**MLP ≈ 3.2 min**, TabResNet ≈ 4 min, CNN ≈ 4 min (GPU), FT-Transformer ≈ 20–30 min (GPU).

| Stage | Runs | Serial CPU-hours | GPU-hours | Wall-clock @ 8 workers | @ 16 workers |
|---|---|---|---|---|---|
| 1. Benchmark (diabetes) | 60 | 3.2 | 0 | 25 min | 12 min |
| 1. Factorial | 40 | 2.1 | 0 | 16 min | 8 min |
| 2. Period sweep | 120 | 6.4 | 0 | 48 min | 24 min |
| 2. Loss sweep | 80 | 4.3 | 0 | 32 min | 16 min |
| 2. Compression | 60 | 3.2 | 0 | 24 min | 12 min |
| 2. Privacy (DP + inversion) | 40 | 2.3 | 0 | 18 min | 9 min |
| 2. Scaling (10→100 clients) | 48 | 5.0 | 0 | 38 min | 19 min |
| 3. eICU benchmark | 60 | 8–12 | 0 | 1.5 h | 45 min |
| 3. MedMNIST (CNN) | 60 | — | **4–6** | 4–6 h GPU | — |
| 3. Transformer arm | 30 | — | **10–15** | 10–15 h GPU | — |
| **Total** | **~600** | **~35–40 CPU-h** | **~15–20 GPU-h** | — | — |

**Bottom line: budget ≈ 15–20 GPU-hours and ≈ 35–40 CPU-core-hours.**
On one modern GPU box (say 16 cores + 1 GPU) the whole programme is
**roughly 1.5–2 days wall-clock**, of which:

* Stage 1 (what decides the paper) — **under 1 hour**;
* Stages 1+2 (a complete, defensible tabular paper) — **~4 hours**;
* Stage 3 (imaging + transformer + eICU, the Q1 differentiators) — the remaining
  15–20 GPU-hours.

If you must economise, cut the transformer arm to 5 seeds and MedMNIST to a
single collection: that halves GPU time to ~8–10 hours with little loss of
argument.

## 5. Practical notes

* Always pass `--threads 1` when running many jobs in parallel (the launcher
  does this); otherwise torch oversubscribes cores and everything slows down.
* Runs are independent and resumable — each writes its own
  `results/logs/*_history.csv`, so a crashed job can simply be relaunched.
* Watch RAM at 100 clients: each client holds its own optimiser state.
  ~8 GB is comfortable for the tabular arms.
* Set `PYTHONHASHSEED=0` for byte-identical reproducibility (`set_seed` already
  sets it inside the process).

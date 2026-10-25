# TierFed v3 — budget-constrained, divergence-driven federated aggregation over constrained links

Code, experiment drivers and manuscript source for

> **Communication as the binding constraint: budget-optimal multi-rate federated learning for clinical prediction over bandwidth-limited links**

v3 is a full rewrite after an internal audit of v2 found reporting defects
(see `ROADMAP.md`, section 0). **Every number in the manuscript is generated
from per-seed logs by `src/analysis/collect.py`**; the LaTeX source contains
no hand-typed results, and tables without runs render `[RUN REQUIRED]`.

## What is here

```
src/
  server.py             federated loop; per-client DP for EVERY strategy; attack hooks; network accounting
  client.py             local training (Adam by default, SGD arm available)
  strategies/           FedAvg FedProx SCAFFOLD(real control variates) FedPer FedLAMA(reimpl.)
                        AdaComm LAG FixedSchedule TierFed(exact integer programme)
  models/               MLP / TabResNet / FT-Transformer / CNN + automatic tiering
  data/                 diabetes (patient-dedup, patient-grouped splits, share-capped specialty partition),
                        eicu (true hospitalid), mimic, medmnist (PneumoniaMNIST), synthetic
  privacy/              DP-FedAvg with exact Gaussian RDP accountant; update-inversion attack on real
                        transmitted slices; loss-threshold membership inference
  network.py            bits -> round time / energy / window compliance under 5 link profiles
  theory/               per-tier dispersion tracking
  analysis/             stats (BCa CIs, Holm, non-inferiority), collect -> LaTeX, figures
  experiments/          run_one / run_benchmark / run_ablation / run_compression / run_privacy / run_scaling
scripts/                setup_env.sh, reproduce_all.sh, check_determinism.sh, claim_check.py
paper/                  main.tex (IEEEtran), sections/, generated/ (from collect.py), supp.tex
docs/convergence.md     the bound, its assumptions, and where the code departs from it
tests/                  unit + integration tests, including bit-identical determinism and
                        "DP is applied to every strategy"
```

## Install and verify

```bash
bash scripts/setup_env.sh venv      # torch 2.7.1 cu128 (RTX 5080 / Blackwell), pinned deps
source .venv/bin/activate
python -m pytest tests -q           # all green before any run
bash scripts/check_determinism.sh   # repeated runs must be bit-identical
```

## Reproduce the paper

```bash
bash scripts/reproduce_all.sh       # stages 1..7, resumable; STAGES="2 3" to select
make paper                          # collect -> tables, figures, claim check, PDF
grep -c "RUN REQUIRED" paper/main.pdf   # must be 0
```

Datasets: `diabetes` downloads automatically (UCI id 296). `medmnist` downloads
on first use. `eicu`/`mimic` need credentialed PhysioNet access; place
`patient.csv` / `admissions.csv` under `data/eicu/` and `data/mimic/` and the
eICU stage runs automatically.

## Reporting rules we hold ourselves to

* Non-inferiority is a one-sided test at a stated margin on **both** AUROC and
  AUPRC; it is never inferred from a non-significant difference test.
* Upload is what the compressor emitted for the clients that actually sent.
* Personalised methods are scored on patient-grouped per-client held-out
  splits; clients with a single class in their held-out split are excluded
  identically for every method and their count is reported; a pooled AUROC
  over all held-out predictions is reported alongside.
* Baselines share one architecture table (`src/experiments/runner.ARCH`).
* DP: per-client clipping, noise on the sum, accountant charges only
  transmitting rounds; epsilon is reported as computed, however large.
* Attacks run on the real transmitted update of every strategy.
* Unreported arms do not exist: every configuration run by
  `reproduce_all.sh` appears in a table or in the supplement.

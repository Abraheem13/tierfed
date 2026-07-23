# Nested Federated Learning (NFL) — v2.0

Multi-frequency, layer-wise aggregation for federated clinical prediction.

**v2.0 is a full rewrite** built to answer the reviewer objections raised on the
v1 manuscript. Every objection maps to code you can run.

---

## Reviewer objection → what was added

| # | Objection | Response in this release |
|---|-----------|--------------------------|
| 1 | *FedPer and FedLAMA omitted — "invalidates the empirical claims"* | Both implemented as first-class strategies (`src/strategies/baselines.py`), plus SCAFFOLD. FedLAMA uses the real adaptive discrepancy-based interval assignment, not a stand-in. |
| 2 | *n = 3 seeds gives no statistical power* | Default is **10 seeds**. Added BCa bootstrap CIs, Hedges' *g*, **Holm–Bonferroni** correction over the whole comparison family, and a rule that suppresses Wilcoxon when n < 6 instead of printing an uninformative p = 0.25. |
| 3 | *PPD is an unorthodox metric that hides non-convergence* | PPD is now reported **alongside** standard statistics: final, best, mean-over-rounds, last-10 mean, and **simulated patience-based early stopping** (`TrajectoryStats`). If the effect only exists in PPD, these will say so. |
| 4 | *No convergence analysis* | `docs/convergence.md` gives a blockwise local-SGD bound with the multi-rate drift term `4η²L²Σ K_g²Γ_g`. The bound's key assumption (per-tier divergence ordering) is **measured every round** by `src/theory/divergence.py`, making the theory falsifiable rather than decorative. |
| 5 | *Partition hardcoded to a 2-hidden-layer MLP* | `src/models/tiering.py` assigns tiers automatically for **any** `nn.Module` via execution-order depth cuts, with an explicit-map fast path. Verified on MLP, TabResNet, FT-Transformer and CNN. |
| 6 | *Single outdated tabular dataset; pseudo-hospitals are synthetic* | Five corpora: `diabetes`, **`eicu` (true `hospitalid`)**, **`mimic`**, **`medmnist` (imaging)**, `synthetic` (controlled covariate/concept shift). Partitioning supports natural sites, Dirichlet label skew, quantity skew, LPT and IID — each with a heterogeneity report. |
| 7 | *Bandwidth saving never benchmarked against compression* | `src/compression.py` implements 8/4-bit quantisation, top-k, sparse ternary (STC) and Count-Sketch with **exact bit accounting**. `run_compression.py` reports NFL *versus* and *composed with* each. |
| 8 | *Cannot separate private head from multi-rate schedule* | The two mechanisms are independent flags, giving a **2×2 factorial**: FedAvg / FedPer / NFL-Sched / NFL. This is the decisive attribution experiment. |
| 9 | *Only inverse-frequency reweighting for 11.4 % prevalence* | Added **focal**, **LDAM**, **class-balanced** losses and a **balanced sampler**; `--kind loss` sweeps them. **AUPRC** and threshold-optimised F1 are now primary metrics. |
| 10 | *No privacy mechanism or evaluation* | Client-level **DP** (clipping + Gaussian) with an RDP accountant reporting (ε, δ), and a **gradient-inversion attack** measuring what an adversary actually recovers per strategy. |
| 11 | *No computational overhead / scalability discussion* | `run_scaling.py` measures wall-clock per round, peak GPU memory and accuracy for 10→100 clients. |

---

## Install

```bash
pip install -r requirements.txt
make test          # 31 unit + integration tests
```

## Reproduce the paper

```bash
make benchmark     # all strategies x 10 seeds  -> tables/benchmark_*
make ablation      # factorial, periods, architectures, losses
make compression   # NFL vs / with quantisation, top-k, STC, sketching
make privacy       # DP utility curve + inversion attack
make scaling       # overhead and client-count scaling
make figures       # regenerate every figure
```

Single run:

```bash
python -m src.experiments.run_benchmark --dataset diabetes --strategies fedavg fedper fedlama nested \
    --seeds 42 43 44 45 46 47 48 49 50 51 --rounds 60 --device cuda
```

## Datasets

`diabetes` downloads automatically (UCI id 296). `eicu` and `mimic` require
credentialed PhysioNet access — download them yourself and point
`NFL_DATA_ROOT` at the folder; the loaders never attempt to fetch restricted
data. `medmnist` downloads on first use. `synthetic` needs nothing and is what
CI runs on.

## Layout

```
src/
  data/         corpora + partition schemes with heterogeneity diagnostics
  models/       MLP / TabResNet / FT-Transformer / CNN + automatic tiering
  strategies/   FedAvg FedProx FedPer FedLAMA SCAFFOLD NestedFL(2x2)
  privacy/      client-level DP (RDP accountant) + gradient-inversion attack
  theory/       per-tier divergence tracking and bound terms
  analysis/     bootstrap CIs, Holm correction, non-inferiority, figures
  experiments/  benchmark / ablation / compression / privacy / scaling
docs/convergence.md   the analysis behind the multi-rate schedule
```

## Notes on honesty

* Peak-AUROC parity is tested as **non-inferiority**, not inferred from a
  non-significant difference test.
* Upload figures come from what the compressor actually emitted, not a nominal
  parameter count.
* Personalised methods are evaluated on **per-client held-out splits**; a single
  pooled test set cannot measure a per-client head.
* The DP module reports a real (ε, δ) budget; it does not claim GDPR adequacy.

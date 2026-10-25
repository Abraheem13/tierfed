# TierFed v3 — execution path to a Q1 submission

This file is the contract for the rewrite. Every item below maps to a code
change, a run on the RTX 5080, or a paper section. **No number enters the
manuscript by hand**: `src/analysis/collect.py` writes `paper/generated/*.tex`
from the per-seed logs, and the manuscript `\input`s those files. Until a run
exists, the corresponding table renders a visible `[RUN REQUIRED]` marker.

## 0. Why a full rewrite (what the audit found)

| # | Defect in v2 | Fix in v3 |
|---|---|---|
| 1 | `Strategy.aggregate` never applied DP for FedAvg/FedPer/FedProx; TierFed DP crashed on local-only rounds | Per-client clipping in the server loop for **every** strategy; empty-round guard; crash reproduced by a unit test |
| 2 | DP clipped the *aggregate*, so (ε,δ) was invalid | Per-client clip, noise σC on the sum, exact Gaussian RDP accountant |
| 3 | SCAFFOLD was FedAvg with 2× bit accounting | Real control variates (option II), local gradient correction, server-side c update |
| 4 | FedLAMA was a rank-quartile ladder | Cumulative discrepancy-share rule of Lee et al. (2023) |
| 5 | Scaling table had no TierFed runs and extrapolated numbers | `run_scaling.py` runs TierFed; LPT cannot give 100 clients so scaling uses Dirichlet partition |
| 6 | Architecture sweep compared different depths | Single `ARCH` dict shared by every strategy |
| 7 | Imaging: binarised 97%-positive DermaMNIST, 1-sample clients, 4/5 seeds | PneumoniaMNIST (natively binary), Dirichlet with `min_size`, clients with one class excluded *for every method*, pooled AUROC also reported |
| 8 | Inversion attack ran on the fixed schedule at init | Attack on the **actual post-training update** of each strategy at every synchronisation round; TierFed included |
| 9 | Paper described a 93-input / 36,930-param MLP; runs used 162 / 45,762 | Tier report is written from the probe model built on the real data; paper reads it from `generated/` |
| 10 | Nondeterminism (identical configs differ by up to 0.006 AUROC) | `torch.use_deterministic_algorithms`, single thread, seeded loaders, `test_determinism` |
| 11 | No patient dedup; one "missing specialty" client held 49% of data | Patient-grouped split; LPT with a client-share cap |
| 12 | Controller restricted to powers of two | Unit-step greedy over [1, K_max]; realised budget reported |
| 13 | Theory used Γ (gradients) but code measured parameter dispersion | Bound restated in terms of the measured quantity; local-step count τ_k enters explicitly; Adam stated as the optimiser with the bound given for SGD |
| 14 | Related work omitted AdaComm, LAG, LG-FedAvg, FedBABU, partial personalisation | All added; AdaComm and LAG implemented as baselines |

## 1. New framing (communications / 6G / network security)

Target venues: IEEE Transactions on Network and Service Management, IEEE
Transactions on Network Science and Engineering, or IEEE Open Journal of the
Communications Society. The clinical application stays as the use case; the
contribution is stated as a **network-side service**:

* The federated job is a tenant on a bandwidth-constrained, scheduled uplink
  (hospital WAN, 5G/6G edge slice, or satellite backhaul). ρ is a service-level
  quota the operator sets, not an ML hyper-parameter.
* `src/network.py` converts measured bits into **round time, job completion
  time, energy and window compliance** under published link profiles
  (3GPP NR uplink, 6G sub-THz edge slice, LEO satellite). Every strategy is
  reported on the same link models.
* Security is evaluated as **attack surface**: bytes exposed per round,
  gradient-inversion success on the real transmitted update, membership
  inference on the released model, and DP with a correct accountant.
* The controller is presented as a budget-constrained scheduler that
  minimises a convergence-bound surrogate, in the language of
  resource-allocation. With two or three shared tiers and K_max = 8 the
  integer programme has at most 512 feasible points, so v3 solves it
  **exactly by enumeration** every round; the greedy rule is kept only as a
  documented fallback for many-tier models.

## 2. Code work (done in this commit)

See `git log` and the file table in README. All v2 result logs were removed
because they were produced by the buggy code path.

## 3. Runs on the RTX 5080 (you do this)

```bash
cd tierfed
bash scripts/setup_env.sh venv          # pins torch 2.4 / CUDA 12.4; CPU fallback documented
source .venv/bin/activate
python -m pytest tests -q               # must be all green before any run
bash scripts/reproduce_all.sh           # stages 1–6, ~20–30 GPU-hours total; resumable
```

Stage order and what each produces (`results/logs/*_summary.json` per run):

| Stage | Script | Runs | Produces |
|---|---|---|---|
| 1 Determinism | `scripts/check_determinism.sh` | 4 | assert identical AUROC for repeated configs |
| 2 Main benchmark | `run_benchmark.py` (diabetes, eicu if available) | 9 methods × 10 seeds | Table main, Holm tests, NI tests on AUROC **and** AUPRC |
| 3 Budget sweep + compression | `run_ablation.py --kind budget`, `run_compression.py` | ρ ∈ {0.15,0.25,0.4,0.6,0.8} × 10; 6 compressors × 2 × 10 | Frontier, composition table |
| 4 Architecture + imaging | `run_ablation.py --kind architecture`, `run_benchmark.py --dataset medmnist` | 4 archs × 4 methods × 5; 4 methods × 10 | Generalisation table |
| 5 Privacy + security | `run_privacy.py` | 4 σ × 4 methods × 10; inversion on every sync round; MIA | DP table, attack table |
| 6 Scaling + network | `run_scaling.py`, `src/network.py` | 4 sizes × 4 methods × 3 | Overhead table, link-model table |

Then:

```bash
python -m src.analysis.collect --results results        # tables/*.csv + paper/generated/*.tex
python -m src.analysis.make_figures --results results   # paper/figures/*.pdf
cd paper && make                                         # main.pdf; grep for RUN REQUIRED
```

If `grep -c "RUN REQUIRED" paper/main.pdf` is not zero, a table is missing a run.

## 4. Paper fill-in rules

* Text statements with numbers use macros from `paper/generated/numbers.tex`
  (e.g. `\numTfVsFedPerDiff`). Never type a number.
* Claims are written conditionally where the outcome is unknown, and
  `scripts/claim_check.py` fails the build if a conditional claim's condition
  is false in `numbers.tex` (e.g. "non-inferior" when p ≥ 0.05).
* Every table caption states n seeds, the test, and the correction.

## 5. Submission checklist

- [ ] All stages green; `claim_check.py` passes
- [ ] Figures regenerated from `results/`
- [ ] Supplementary: full proof (`docs/convergence.md` → `paper/supp.tex`), DP accounting, per-seed tables
- [ ] Data availability, funding, CoI, ethics statements present
- [ ] Prior-submission disclosure in cover letter
- [ ] Page limit of the chosen venue checked

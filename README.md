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

## v4 experiments (matched-epsilon DP, real institutions, wide model + link emulation)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_v4.ps1          # all parts; -Part emu|dp|real for one
powershell -ExecutionPolicy Bypass -File scripts\check_v4.ps1        # progress
```

* `run_dp_matched`: FedAvg, FedPer, fixed period and TierFed-DP at the same target epsilon,
  Poisson client sampling, sigma calibrated per method (subsampled-Gaussian RDP accountant).
  Data-dependent schedulers (adaptive TierFed, AdaComm, LAG, FedLAMA, SCAFFOLD) are refused under DP.
* `run_realsites`: UCI Heart Disease (4 institutions) and the open eICU-CRD demo (one client per hospital).
  Both download once into `data/`. Full eICU-CRD: put `patient.csv` in `data/eicu` (credentialed).
* `run_emulation`: ~1.8M-parameter `mlp_wide`; every run's logged per-client upload bytes are replayed
  over paced loopback TCP (`src/netemu.py`) for time-to-target. Application-layer, not netem.

## v6: reproduce every number of the paper in one command

Every v6 run splits each site's patients 70/10/20 into train, validation and test. The
reported AUROC is the test AUROC at the round with the highest validation AUROC; test
data never choose anything. Tags start with `_au_`, so earlier logs stay reproducible.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_cpu.ps1                 # once (CPU-only machines)
powershell -ExecutionPolicy Bypass -File scripts\reproduce_v6.ps1 -Part all -Jobs 3
```
```bash
bash scripts/reproduce_v6.sh all 4
```
Parts: `core` (main table, bounds, frontier, DP at q = 1, peak), `gate` (heterogeneity sweep,
timing side channel), `resnet` (backbone test), `analysis` (consistency and cross-machine checks,
tables, figures, numbers, and the PDFs when `pdflatex` is installed). Every part except
`analysis` first runs `scripts/check_determinism` (one configuration twice, bit-identical or stop).
Finished runs are skipped, so the script can be stopped and restarted.
`analysis` alone regenerates every table, figure and number from the logs in `results/logs`
without training anything. The plan is 405 runs; on the 4-core Linux machine used for the paper they took about three hours, and `analysis` takes about 15 minutes.

`src/analysis/consistency.py` recomputes every transmitting-round count, upload volume
and DP charge from the schedule rules and asserts them against the logs.
`src/analysis/crossmachine.py` compares every run in `results/crossmachine/<machine>/logs`
(the heterogeneity and timing runs repeated on a Windows laptop, see its `machine.json`)
with the run of the same name in `results/logs`: configuration, bits per round and site,
planned periods, selected round and AUROC. It also recomputes the gate decision and the
timing verdicts from the second machine's runs alone.
`src/analysis/audit_report.py` writes the decision gate, the iso-byte budgets and every
audit table to `results/tables/audit/`; `src/analysis/audit_tex.py` and
`src/analysis/audit_figures.py` write the manuscript's tables, numbers and Fig. 5;
`src/analysis/results_figures.py` draws Figs. 4 and 6 from the per-round histories.
`src/theory/differentiation.py` holds the closed form of the interval programme and the
differentiation bound Psi (Proposition 2); `src/privacy/gdp.py` the exact q = 1 accountant.

## Manuscript figures and tables

```powershell
.\.venv\Scripts\python.exe -m src.analysis.collect --results results --dataset diabetes --model mlp
.\.venv\Scripts\python.exe -m src.analysis.paper_figures --results results --out paper\figures
```
`collect` writes every table body and number macro under `paper/generated/`; `paper_figures` draws the
multi-panel figures of the manuscript from the per-seed histories. The flowcharts are TikZ in the LaTeX source.

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

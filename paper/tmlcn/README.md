# TMLCN submission package

Manuscript for the IEEE Transactions on Machine Learning in Communications and Networking:
"When Does Per-Tier Synchronisation Pay? Quota-Constrained Federated Learning Over Constrained Links".

| File | What it is |
|---|---|
| `main.tex`, `main.pdf` | the article (IEEEtran, journal mode), self-contained; numbers written as plain text |
| `supp.tex`, `supp.pdf` | the supplementary material |
| `figures/` | every figure both files include (PDF) |
| `references.bib` | bibliography database |
| `cover_letter.md` | cover letter; fill the two items marked **[CONFIRM]** before sending |
| `REFERENCE_CHECK.md` | what was checked for each cited reference, and what was not |

## Build

```
pdflatex main && bibtex main && pdflatex main && pdflatex main
pdflatex supp && bibtex supp && pdflatex supp && pdflatex supp
```

## Where the numbers come from

Every number in `main.tex` and `supp.tex` is either a value written from the per-seed logs by
`src/analysis/audit_tex.py` (in `paper/generated/`) or an output of the revision scripts in
`scripts/tmlcn/`, which read only `results/logs`:

- `selection_table.py` → `selection_table.out`: both primary contrasts under three round-selection rules (supplement, Section S4).
- `dp_local_paired.py` → `dp_local_paired.out`: the DP arms against local training on the same six seeds (Table VIII).
- `check_numbers.py main.tex`: lists every number in the manuscript that is not among those generated values; the remainder are definitions and constants (thresholds, configuration sets, the e-mail address).

## Open evidence gaps

`scripts/tmlcn/run_new_experiments.sh` runs the two experiments the manuscript states as limits:
the ResNet backbone at ten seeds, and the timing side channel on the ResNet, where TierFed adapts,
with twenty runs per world. Both need `data/raw/diabetic_data.csv`.

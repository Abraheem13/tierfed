# TMLCN manuscript

"Per-Tier Synchronisation in Communication-Budgeted Federated Learning: A Closed-Form Bound
and an Empirical Study", for the IEEE Transactions on Machine Learning in Communications and
Networking.

| Path | Content |
|---|---|
| `main.tex`, `main.pdf` | the article, self-contained (IEEEtran, journal mode); written by the build |
| `supp.tex`, `supp.pdf` | the supplementary material; written by the build |
| `src/main.tex`, `src/supp.tex` | the source of both: prose without numbers (`\num`, `\ifflag`, `\claim`, `\gen`) |
| `generated/` | numbers, flags and tables written from the run logs by `src/analysis/tmlcn_tex.py` |
| `figures/` | figures written from the run logs |
| `references.bib`, `REFERENCE_CHECK.md` | bibliography, and what each entry was checked against |
| `response_to_review.md` | point-by-point response to the review of the previous version |
| `cover_letter.md` | cover letter |
| `submission/TierFed_TMLCN_submission.zip` | the files to upload: `.tex`, `.bib`, `.bbl`, figures and both PDFs |

## Build

```
python scripts/tmlcn/build.py              # flatten the source, compile, check
python scripts/tmlcn/build.py --analysis   # first regenerate every number, table and figure from results/logs
python scripts/tmlcn/build.py --zip        # also write the submission archive and test that it compiles alone
```

The build stops when a sentence of the source is not supported by the logs: every statement
that depends on an outcome carries a `\claim{flag}` whose flag the analysis computes. It also
stops on an undefined reference or citation, a BibTeX warning, a placeholder in the PDF, or a
font that is not embedded.

`main.tex` and `supp.tex` in this folder compile on their own:

```
pdflatex main && bibtex main && pdflatex main && pdflatex main
pdflatex supp && bibtex supp && pdflatex supp && pdflatex supp
```

## Runs

`scripts/reproduce_v6.sh` (Linux, macOS) and `scripts/reproduce_v6.ps1` (Windows) run every
experiment of the article and then this build. Parts: `core`, `gate`, `resnet`, `centres`
(four clinical centres), `sync` (carry-over synchronisation step), `analysis`, `all`.
Finished runs are skipped.

# TierFed manuscript for IEEE Open Journal of the Communications Society (OJ-COMS)

Title: Quota-Constrained Multi-Rate Synchronization in Federated Learning: Gain Bound, Peak-Load Control, and Schedule Leakage

## 1. Status

- Format: IEEE Open Journals class (IEEEoj.cls). main.pdf 18 pages (25 or fewer = "Original Manuscript"; the journal has no page limit), supp.pdf 8 pages, 57 + 18 references, abstract exactly 250 words.
- Build: 0 LaTeX errors, 0 undefined references, 0 BibTeX warnings. The log shows 28 "Overfull \vbox ... while \output is active" messages; they come from the page header of the official class and are not text overflows.
- Checks in this round: compliance and layout (0 blockers, 2 major, 13 minor), hostile language read of the article (verdict SEND TO REVIEW; 0 blockers, 0 major, 7 minor), supplement and references (0 blockers, 0 major, 6 minor). All exact fixes applied. Earlier rounds are in audit_reports/.
- Confirmed on the journal's pages on 26 Nov. 2026: Editor-in-Chief Sinem Coleri (Koc University); topic areas "Big Data and Machine Learning for Communications" and "Cloud Computing, Edge Computing, and Internet of Things"; single-anonymous review with at least two reviewers; no page limit; APC US$2,160 paid after acceptance. Per the template, the abstract has no abbreviations, footnotes or references and is one paragraph of 250 words; index terms are alphabetical.
- Exhaustive verification round (six agents, one ledger row per item; ledgers in audit_reports/R5):
  - Article numbers: 1,289 items; 1,255 verified against logs/code, 27 depend on bootstrap order, 7 wrong (4 sentences that gave seed-42 values as if general; now give the range over seeds), 0 unverifiable. All 588 numeric table cells recomputed from the logs.
  - Supplement numbers: 1,753 items; 1,752 verified, 1 wrong (Table S-XII caption: 16 to 20 of the 20 clients hold data), 0 unverifiable. Proof algebra checked in 14 steps.
  - References: 514 fields of 59 entries checked at opened sources; 511 verified, 2 corrected (an author initial; a conference city), 1 unverifiable (the "3rd Quart." label of the Lim et al. survey; volume, issue, pages, year confirmed).
  - Citations: 112 citation-sentence pairs; 101 supported, 10 narrowed or moved, 1 unverifiable (Efron 1987 full text). Notable: the text now says FLamby retains 740 Fed-Heart-Disease patients while we keep all 920; gradient boosting is scikit-learn's HistGradientBoostingClassifier, not LightGBM.
  - Sentences: 1,185 items (841 sentences, 26 equations, 318 table rows); no grammar defect; 13 notation or consistency defects fixed.
  - BOOTSTRAP ORDER FIXED IN THE REPOSITORY: src/analysis/audit_report.py now reads the logs in sorted order and orders the runs by seed, so the seeded bootstrap no longer depends on the directory order. The tables were regenerated. Changed in the paper: the interval limits of Table 3 (10 rows), Table 4 (10 rows) and Table S-II (5 rows), four intervals quoted in Sections VI-D and VI-E, and the permutation p-value of LAG with xi = 2000 in Table 9 and Section VI-G (0.009 to 0.010). No mean, difference, upload, count or other p-value changed, and every generated table row now has an identical row in the paper (script check). LAG and FedAvg in Table 4 now share one interval, as their identical per-seed values require.
- Closing pass: citations added and verified at source for the residual network (He et al., CVPR 2016), Dirichlet label skew (Hsu et al., arXiv:1909.06335), ECE (Guo et al., ICML 2017) and the Brier score (Brier, 1950); the article now has 57 references. The dataset paragraph states that Strack et al. retain 69,984 patients with the cleaning steps in the opposite order. The Wilcoxon and Holm procedures are standard and remain uncited.
- This covers language, references and journal format. It is not a prediction of acceptance (Section 5).

## 2. What is in this package

- main.pdf: the manuscript to upload.
- supp.pdf: supplementary material.
- submission/gagraphic.png and gagraphic_caption.txt: the required graphical abstract (660 x 295 px, drawn from Table 5) and its caption.
- submission/figures/rashi1.pdf ... rashi7.pdf: the seven figures as separate files, named in the IEEE convention.
- submission/cover_letter.txt: cover letter with the prior-rejection disclosure that OJ-COMS requires.
- main.tex, supp.tex, references.bib, main.bbl, supp.bbl, IEEEoj.cls, ojcoms.png, figures/: LaTeX sources.
- audit_reports/: every report of the four checking rounds.

Submission portal: https://ieee.atyponrex.com/dashboard?siteName=oj-coms

## 3a. Changes made for OJ-COMS

- Converted to the Open Journals template: front matter, author and affiliation block, corresponding-author line, uppercase headings, running head.
- The article's tables are now numbered 1-9 (Arabic) by the class; the supplement's pointers to them were updated and checked.
- "Data, Code, and Declarations" split into "Data and Code Availability" and "Acknowledgment"; the Acknowledgment carries the AI-use disclosure that IEEE requires.
- Adam is cited at first mention; the Conclusion carries the "in nearly every round" qualifier used in the abstract; small wording and hyphenation fixes.
- Supplement: citations sorted, signs added to one column of Table S-XIV and its caption made explicit, postcodes added.

## 3b. Corrections of substance made since the first submission (please read; these are not language edits)

The audit against the logs and code found statements in the submitted text that were wrong or unsupported. Each was corrected to what the logs and code show.

1. The only changed number: clinical centers, TierFed minus FedPer, interval [-0.0158, -0.0009] in the main text is now [-0.0161, -0.0010], the value in Table S-I. The two came from the same bootstrap read in a different seed order. Confirm which you want and regenerate both from one call.
2. Five numbers added from the logs: final-round AUROC of TierFed 0.6289, staggered 0.6269, AdaComm 0.6334, FedPer 0.6238, Fixed (8,8) 0.6159. The submitted sentence said a final-round comparison would favor the sparse schedules; that is false for (8,8).
3. Fractions 0.04, 0.88, 0.34, 0.93 are fractions of all 60 rounds (the code divides by 60), not "of the rounds after warm-up".
4. "At most 2.2%" is the largest mean over seeds; the text now says "mean".
5. The 100-client sentence no longer says "a larger federation alone did not change this"; there is no local-training reference for that experiment.
6. Abstract: "showed no gain ... in seven settings at matched bytes" (Table V), and the leaking baseline is named.
7. Fig. 3 caption: the simulator does log the dispersion of private slices as a diagnostic; only the scheduler is shielded.
8. "The released code refuses to run these under DP" now mentions the override flag.
9. Label flip: each label is flipped with probability 0.5, not exactly half of them.
10. Table V note lists which uniform periods were run in each setting (only K = 4 on the ResNet).
11. Supplement: Table S-XIII reports peak GPU memory, not CPU; Table S-XII p-values are from a paired t-test, not the signed-rank test; Tables S-XV/S-XVI fixed periods follow an older rule (19 and 12 releases, not 18 and 11); Table S-XIV fixed schedule is (4,4) without warm-up; the Renyi order grid is stated as in the code; the constants c0, c2 of the proof are stated consistently; "FedLAMA" is "FedLAMA-style" in two legacy tables.
12. Table II: "Adam (learning rate 1e-3)" (confirmed in src/experiments/runner.py).

All table cells and all equations are otherwise identical to the submitted version (tools/numcheck.py).

## 4. What only the authors can do before submitting

1. Recompile with the official class file. The structure of main.tex was checked against IEEE's own Open Journals template on Overleaf (\documentclass{IEEEoj}, date and DOI lines, uppercase author names, \affil, \corresp, \authornote, \markboth, abstract and keywords before \maketitle, uppercase section and subsection headings, \IEEEPARstart, "ACKNOWLEDGMENT") and matches it. The IEEEoj.cls file itself was taken from a public repository of an OJ-COMS manuscript, because IEEE's download site could not be reached from here; get the OJ-COMS template from https://template-selector.ieee.org/ and build main.tex with its class file. IEEE Xplore could not be opened from here either, so the look of published OJ-COMS articles was not compared directly.
2. Confirm the Acknowledgment. It names Claude (Anthropic) for language editing of every section and for redrafting the abstract and Sections I and II. The sentence "Generative AI tools were also used for code refactoring" names no system; IEEE policy covers code, so name the tool you used.
3. Confirm that TMLCN is the only journal that has rejected this manuscript. OJ-COMS accepts a previously rejected paper only once and only if it was rejected once; the cover letter says so.
4. Mehwish Mahmood: add her department and confirm the spelling (the TMLCN e-mail used "m.mehmood@qub.ac.uk"). The affiliation line currently reads "Queen's University Belfast, Belfast BT7 1NN, U.K."
5. ORCID is required for all three authors and is entered in the submission system. Add IEEE membership grades if any author is a member. Biographies and photos are optional at submission.
6. Read the article once and confirm that every "we" sentence describes something you did.
7. References: every field is now verified at a source except the "3rd Quart." label of the Lim et al. survey.
8. PneumoniaMNIST is used only in the supplement and is not cited there (it is cited in the article). Adding the citation pushes one reference onto a ninth page.
9. Tables 1, 3, 6 and 9 print at about 6.6-7.2 pt in the narrower text block of this class. They are readable; IEEE will reset them at production.
10d. Editor P. L. Bartlett of the Platt (1999) book: sources disagree on the middle initial (MIT Press prints none; university records give "L."). It is printed "P. L. Bartlett".
10. Repository: the final sources are committed under paper/ojcoms/ on the branch ojcoms-submission. The older macro-based sources under paper/tmlcn/ are the TMLCN version and were not rewritten; do not rebuild the submission from them.

## 5. Weaknesses that editing cannot remove

- The headline result is negative: in five of seven settings the scheduler returns the uniform period it is compared with, and in the other two it does not beat it.
- Fit to a communications journal is thin: the network is a three-parameter link model with no measurements.
- The benchmarks barely resolve the effects: AUROC is about 0.64, differences of 0.001-0.005 are judged against a 0.005 threshold with intervals of similar width; the real-institution data have four clients and 736 patients.
- Five seeds for the label-skew sweep, the frontier and the ResNet carry-over comparison; the second primary criterion was changed after the fact (disclosed).
- Supplement Section S9 uses a different protocol that the article itself calls not comparable.
- The prose is correct but uniform; with a declared use of AI for editing, some reviewers may read it as machine-edited.

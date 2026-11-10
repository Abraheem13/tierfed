# Response to the review of the previous version

Manuscript: "Per-Tier Synchronisation in Communication-Budgeted Federated Learning: A Closed-Form
Bound and an Empirical Study" (previous title: "When Does Per-Tier Synchronisation Pay? A
Closed-Form Test and a Traffic Side Channel in Communication-Budgeted Federated Learning").

All nine comments of the annotated manuscript are answered below, in the order in which they
appear. Section, equation and table numbers refer to the revised article. Every number quoted
here is taken from the same generated values as the article.

---

**Comment 1 (title).** *"The title promises more than the results show."*

The title has been changed to "Per-Tier Synchronisation in Communication-Budgeted Federated
Learning: A Closed-Form Bound and an Empirical Study". It no longer asks when per-tier
synchronisation pays, because the results do not answer that question in the affirmative: in none
of the seven settings was the per-tier schedule more accurate than the best uniform period at
matched bytes. "Test" has become "bound", because Psi bounds the reduction of a surrogate and was
not shown to predict a gain in accuracy, and the side channel is no longer in the title. The
running head, the abstract, the contribution list and the conclusion were rewritten to match.

**Comment 2 (abstract, 65.3 of 217.8 MB).** *"These numbers give a 70.0% saving over 60 rounds. Up
to the selected checkpoint, the saving is 27.7% (45.7 vs 63.2 MB). Say which comparison the
headline uses and keep it consistent with the main metric."*

Both readings are now defined and named, and no saving is stated without its reading.
Section II-D defines the cumulative upload B(r) (Eq. (8)) and its two readings: B(r-hat), the
upload up to the validation-selected round, which is the cost that belongs to the primary metric,
and B(R), the upload of the whole budget of 60 rounds. The abstract now reads "28% less upload up
to that round and 70% less over all 60 rounds" (45.7 against 63.2 MB, and 65.3 against 217.8 MB;
percentages are rounded to integers so that they agree with the rounded megabytes). Section V-B
states both with their megabytes and adds that the scheduler selects round 36 on average and
FedPer round 17, so the saving up to the selected round is obtained with about twice as many rounds.
Tables IV and V and the paired table of the supplement carry both readings in separate columns.

**Comment 3 (Section II-A, one local epoch).** *"Are inactive tiers kept locally or reset from the
server? Figure 2 suggests only active tiers are replaced. Clarify this, including what happens to
the optimizer state."*

Section II-A now states the complete round as three equations (Eqs. (1)-(3)) and says explicitly:

- a tier that is not due is kept at the client, is neither reset nor frozen, and continues to be
  trained; the server copy does not change;
- when a tier is due, the client first replaces its copy by the server copy (Eq. (1)), trains one
  epoch and uploads; after aggregation it adopts the new average (Eq. (3));
- the optimiser state belongs to the client: Adam's moment estimates and step count are kept from
  round to round for all parameters and are not reset when the values of a tier are replaced.

The caption of Fig. 2 and that of Fig. 3 now say the same. Answering this comment exactly also
showed that the previous text described the synchronisation inaccurately in one respect, which
has been corrected throughout. Because a due tier is restarted from the server copy, what a
client uploads is always one local epoch away from the broadcast, whatever the interval. The
previous statements that the received slices "result from (r - t_g) s local steps" and that the
dispersion "can grow quadratically with the interval" described the textbook variant, not the
implementation. The revision therefore

- names the two variants (restart step, used by every experiment; carry-over step, in which a
  tier is uploaded after all local epochs since its last synchronisation);
- restates Proposition 1 for the restart step, with the drift of the clients' copies and the
  bound for the server copy as separate parts, and gives the bound of the carry-over step next
  to it (Section III-B, proofs in Section S1);
- reports the measured dispersion for both: under the restart step it is flat in the period
  (slow tier 4.59-5.93, medium tier 6.93-7.71, times 1e-3, for K = 1 to 8), under the
  carry-over step it is larger by factors of 7.5 and 5.3 at K = 4 and of 26.9 and 11.9 at K = 8
  (Table S-V, Fig. 7(d));
- adds an experiment with the carry-over step on the specialty groups, the clinical centres and
  the ResNet (Section V-E, Table VII, Fig. 7(c)).

The main finding is the same under both steps: the scheduler reduces to the uniform period. The
step does change how a sparse period compares with FedPer, and this is reported (see comment 4).

**Comment 4 (Section II-C, sites from the medical-specialty column).** *"These are specialty-based
groups, not separate hospitals. Make that clear throughout. A real multi-institution dataset
under the same protocol would strengthen the clinical relevance."*

Both parts were addressed.

(a) Wording. Section II-C now says that the public release carries no hospital identifier, that
the clients of the main benchmark "are not hospitals" but 20 groups of patients formed from the
medical-specialty column, and names them specialty groups. The word "site" is no longer used: the
generic participant is a "client", the clients of the readmission benchmark are "specialty
groups", and the clients of the new benchmark are "clinical centres". This applies to the
abstract, all tables, figures and captions, and the limitation is repeated in the Discussion.

(b) Real institutions. Fed-Heart-Disease (the four collections of the UCI Heart Disease data:
Cleveland, Budapest, Zurich/Basel and Long Beach; 920 patients) was run under the protocol of the
main benchmark, with the methods of the main comparison and 20 seeds (Section V-C, Table V,
Fig. 7(a,b), supplement Section S7). The result is reported as it came out:

- the scheduler again selects the uniform period (4,4) in every seed (Psi = 0.7%), and its runs are
  identical to those of that period;
- under the restart step this period is 0.0085 AUROC below FedPer at the selected round
  (p =0.015), which exceeds the relevance threshold, with 5.3 against 9.2 MB of upload up to that round;
- under the carry-over step the same period is level with FedPer (difference +0.0016
  [-0.0067, +0.0140]) with 4.1 MB of upload up to its selected round;
- the dataset is small (the smallest centre has 95 patients), and the article says that the
  result is a check of the behaviour on real institutions and not a clinical evaluation.

**Comment 5 (Proposition 1, sigma_g^2 / N).** *"Why is the variance divided by N?"*

The sentence after Proposition 1 now explains it: every step of the averaged model combines one
stochastic gradient from each of the N clients; these are independent with variance at most
sigma_g^2, so their average with equal weights has variance at most sigma_g^2 / N (the linear
speed-up of local SGD). The proposition now states the equal-weight assumption explicitly, and
the text adds what replaces 1/N under the sample weights used in the experiments: the sum of the
squared weights, i.e. 1/N_eff, with N_eff = 11.8 for the 20 specialty groups and 3.6 for the four
centres. The derivation is in Section S1.2.

**Comment 6 (privacy planner, "can only choose equal intervals").** *"Usually, we will avoid using
words such as 'only', since it is too absolute. It is better to double check if this statement is
correct."*

The statement was checked and was not correct as written, and it has been replaced. Under the
flat prior the relaxed minimiser of the drift term is a uniform period (Proposition 2 gives
Psi = 0), but the integer planner is not restricted to equal intervals. It was evaluated on a grid
of 35 combinations of prior weight and quota (identical for four privacy budgets): it returns
equal intervals in 31 of them, including all those without a binding quota, and the unequal pair
(5,6) in the remaining 4, all at a quota of 0.25, where counting the warm-up makes (4,4) infeasible
(Section III-D, Table S-IV). The corresponding sentence of the Discussion was removed.

**Comment 7 ("The final-round lead of the sparse schedules, reported in an earlier version of this
work").** *"???"*

All references to earlier versions of the work were removed from the article, the supplement and
the declarations. The observation itself is kept and is now stated on the data of the article:
over the last ten rounds the sparse schedules are ahead of FedPer (0.6346 against 0.6261), so a
comparison at the final round would favour them, whereas at the validation-selected round the
difference is -0.0007 (Section V-B). The section of the supplement that listed changes
with respect to earlier versions was removed; supplementary experiments that use a different
configuration are described by that configuration (Section S8).

**Comment 8 (Discussion, "Some conclusions of this study are firmer than others.").** *"??? AI
generated?"*

The sentence was removed. The Discussion now opens by stating its three parts (claims that the
evidence does not support, limitations, recommendations). Beyond this sentence, the whole text
was rewritten in plain technical prose: short aphoristic sentences and contrast formulas of the
kind flagged here were removed throughout, including "Two lessons follow", "It is local training
from a common initialisation, not federation" and "Staggering serves the transfer window, not
energy".

**Comment 9 (caption of the timing table).** *"Look like AI generated table title."*

Every table now has a short descriptive title, and what the columns contain is given in a note
below the table instead of a telegraphic caption. The timing table is titled "Observer AUC of the
Timing Side Channel for Each Schedule and Observable". The figure captions were rewritten as full
sentences.

---

## Further changes made during the revision

- **More runs.** The ResNet comparison now has 6 seeds and the matched-budget DP comparison
  6; the timing experiment has up to 7 runs per world, includes AdaComm, and was repeated on the
  ResNet, where the scheduler adapts. The article reports 855 runs of the main protocol.
- **Timing experiment.** The leak criterion now requires a permutation test (largest deviation of
  the AUC from 0.5 over the four observables, 10,000 permutations) in addition to the AUC threshold.
- **Consistency check.** It was rerun on all 2,218 logs: 0 mismatches in 977 schedule, 872 byte and
  230 accountant checks.
- **References.** All 57 cited entries were re-checked against external records; four were
  corrected and four completed (see `REFERENCE_CHECK.md`).
- **Preprocessing.** The article now states that features are standardised, and missing values
  imputed, with label-free statistics of the pooled data.

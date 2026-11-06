# Cover letter (draft for v13)

Before sending: confirm the two items in square brackets.

Dear Editor,

Please consider the enclosed manuscript, "Every Round Is a Release:
Communication-Budgeted Federated Learning Over Constrained Links", for
publication in the IEEE Internet of Things Journal.

The article treats a cross-silo federated learning job as a tenant on a
constrained uplink with a byte quota, and asks when synchronising different
parts of a model at different rates pays. It contributes:

(i) TierFed, a quota-constrained multi-rate scheduler whose programme has a
closed form. A Hölder bound, measurable at the end of warm-up, limits how much
any per-tier schedule can gain over a uniform period at the same quota, and
phases cap the payload of any single round at an unchanged mean;

(ii) the observation that under client-level differential privacy every
synchronisation is a release, so a schedule computed from client updates is
not charged by the accountant and is visible in the traffic. A timing
experiment shows that an adaptive scheduler's traffic reveals a site with
shifted labels, even when that site never skips an upload;

(iii) an audited evaluation: rounds selected on validation, every baseline
with its own knob swept, local, centralised and gradient-boosted references, a
heterogeneity sweep, a deeper backbone, matched-budget privacy, a consistency
check that recomputes every release and byte from the stated rules, and a
replication on a second machine.

The results are reported as they came out. On the main readmission benchmark
the bound is 0.1% and TierFed reduces to a uniform period. On a ResNet backbone
the bound is 71% and the tiers are split as predicted, but the split does not
beat the best uniform period at matched bytes. Staggered phases reduce
transfer-window violations at an unchanged mean, and under client-level
privacy with 20 sites every releasing schedule falls far below local training.

Prior versions. [Confirm: an earlier version, titled "Nested Federated
Learning" and later "TierFed", was submitted to (journal) and (withdrawn /
rejected / under review).] The defects found in that version are listed in
Section S1 of the supplement. The present manuscript is a full rewrite with
corrected code, and every number in it is generated from released per-seed
logs.

[Confirm the author list and that every listed author has approved the
submission.]

The author declares no competing interests and no funding. All data are public
and de-identified. Code, per-seed logs and the one-command reproduction are at
https://github.com/Abraheem13/tierfed.

Sincerely,
Abraheem Rashid
Department of Computer Science, Brunel University London

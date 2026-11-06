# Cover letter — IEEE Transactions on Machine Learning in Communications and Networking

> Before sending, fill the two items marked **[CONFIRM]**. They concern facts only the author can state.

Dear Editor-in-Chief,

Please consider the enclosed manuscript, "When Does Per-Tier Synchronisation Pay? Quota-Constrained Federated Learning Over Constrained Links", for publication in the IEEE Transactions on Machine Learning in Communications and Networking.

The article treats a cross-silo federated learning job as a tenant on a constrained uplink with a byte quota and asks when synchronising different parts of a model at different rates pays. It contributes:

1. **A closed-form test of the value of per-tier synchronisation.** The relaxed quota-constrained programme of a blockwise local-SGD surrogate has a closed form, and a Hölder bound Ψ, measurable at the end of warm-up, limits how much any per-tier schedule can reduce that surrogate below a uniform period. Phases chosen by a gcd condition cap the payload of a single round.
2. **The schedule as a privacy release.** Under client-level differential privacy every synchronisation is a release, so a schedule computed from client updates is not charged by the accountant and is visible in the traffic. A timing experiment shows that an adaptive scheduler's traffic reveals a site with shifted labels even when that site never skips an upload.
3. **An audited evaluation.** Rounds are selected on validation and the conclusions are re-read under two other selection rules; every baseline has its own knob swept; local, centralised and gradient-boosted references bound the task; a consistency check recomputes every planned release, uncompressed upload and privacy charge from the stated rules; and part of the runs is repeated on a second machine.

The results are reported as they came out, including where they cut against the proposed scheduler. On the main readmission benchmark Ψ is 0.1% and the scheduler reduces to a uniform period. On a ResNet Ψ is 71.4% and the tiers split as predicted, but no reading of the runs shows a relevant gain over the best uniform period at matched bytes. Staggered phases reduce transfer-window violations from 0.30 to 0.08 of rounds at the cost of more radio wake-ups. Under client-level privacy with 20 sites every releasing schedule ends below local training.

We believe the work fits the journal's scope on distributed and federated machine learning for efficient network performance and on the scalability of machine learning in networks, and its emphasis on reproducible sharing of code and artefacts: code, per-seed logs and a one-command reproduction are public at https://github.com/Abraheem13/tierfed.

**Prior versions. [CONFIRM]** An earlier version of this work, titled "Nested Federated Learning" and later "TierFed", was submitted to ______ (journal) and is ______ (withdrawn / rejected / under review). Section S1 of the supplement lists every defect found in that version and how it was corrected; the present manuscript is a full rewrite with corrected code, and no version of it is under review elsewhere.

**Authorship. [CONFIRM]** The listed author is the sole author. Earlier drafts of this work listed co-authors; each has agreed in writing to the present author list. (Replace this paragraph with the correct author list and approvals if that is not the case.)

The author declares no competing interests and no funding. All data are public and de-identified; no ethics approval was required. Generative AI tools assisted with code refactoring, figure layout and editing; the author verified all content, code, numbers and figures.

Sincerely,
Abraheem Rashid
Department of Computer Science, Brunel University London

# Cover letter (draft)

Dear Editor,

Please consider the enclosed manuscript, "Communication as the Binding
Constraint: Budget-Optimal Multi-Rate Federated Learning for Clinical
Prediction over Bandwidth-Limited Links", for publication in IEEE Transactions
on Network and Service Management.

The article treats a cross-silo federated learning job as a network tenant
with an explicit communication quota and asks how that quota should be spent.
It (i) shows that the common heuristic of synchronising early layers less
often is harmful on heterogeneous clinical data when fixed in advance, (ii)
explains this with a blockwise local-SGD bound, (iii) proposes a controller
that solves the resulting budget-constrained integer programme exactly from
online measurements, and (iv) evaluates every method on network-level cost
under published link profiles and on attack surface (update inversion on the
transmitted slices, membership inference, client-level differential privacy).

Disclosure of a prior submission. An earlier version of this work, titled
"Nested Federated Learning" and later "TierFed", was submitted to the IEEE Open
Journal of Engineering in Medicine and Biology and is withdrawn. An internal
audit found reporting defects in that version (listed in Section S1 of the
supplement); the present manuscript is a full rewrite with corrected code, a
deterministic pipeline, and every number generated from released per-seed
logs. No part of the earlier text or results is reused.

The author declares no competing interests. All data are public or
credentialed (eICU) and de-identified. Code and logs are at
https://github.com/Abraheem13/tierfed.

Sincerely,
Abraheem Rashid

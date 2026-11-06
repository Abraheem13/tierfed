"""Paired contrasts under three readings, with bootstrap 95% intervals.

For every setting of the gate (specialty sites, Dirichlet label skew, ResNet)
this prints TierFed minus FedPer and TierFed minus the best uniform period of
Table VI (the period K_u chosen there on validation at matched bytes), under
  argmax  : the paper's rule, first argmax of A^V
  smooth3 : first argmax of the trailing three-round mean of A^V
  last10  : mean A^T over the last ten rounds (no selection)
Intervals are percentile bootstrap over seeds (10,000 resamples, numpy seed 0),
as in the paper. Reads only results/logs; output is pasted into Table S-sel.
"""
import glob
import sys

import numpy as np

sys.path.insert(0, "scripts/tmlcn")
from selection_robustness import readings  # noqa: E402

LOG = sys.argv[1] if len(sys.argv) > 1 else "results/logs"
RULES = ("argmax", "smooth3", "last10")

# setting: (model, run tag prefix, best uniform stem from Table VI, seeds)
SETTINGS = [
    ("Specialty sites", "mlp", "", "fixed44", range(42, 52)),
    ("Dirichlet a=0.1", "mlp", "het_a0.1_", "fixed88", range(42, 52)),
    ("Dirichlet a=0.3", "mlp", "het_a0.3_", "fixed44", range(42, 52)),
    ("Dirichlet a=1", "mlp", "het_a1_", "fixed44", range(42, 52)),
    ("Dirichlet a=10", "mlp", "het_a10_", "fixed44", range(42, 52)),
    ("ResNet backbone", "resnet", "bb_", "fixed44", range(42, 52)),
]


def path(model, strat, tag, stem, s):
    hits = glob.glob(f"{LOG}/diabetes_{model}_{strat}_R60_C*_s{s}_au_{tag}{stem}_history.csv")
    return hits[0] if len(hits) == 1 else None


def boot(d, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(d)
    m = rng.choice(d, size=(n, d.size), replace=True).mean(1)
    return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def main():
    for name, model, tag, uni, seeds in SETTINGS:
        tf, fp, un = {}, {}, {}
        for s in seeds:
            a = path(model, "tierfed", tag, "tierfed", s)
            b = path(model, "fedper", tag, "fedper", s)
            c = path(model, "fixed", tag, uni, s)
            if a and b and c:
                tf[s], fp[s], un[s] = readings(a), readings(b), readings(c)
        S = sorted(tf)
        for rule in RULES:
            d1 = [tf[s][rule] - fp[s][rule] for s in S]
            d2 = [tf[s][rule] - un[s][rule] for s in S]
            m1, l1, h1 = boot(d1)
            m2, l2, h2 = boot(d2)
            print(f"{name:16s} n={len(S):2d} {rule:8s} TF-FedPer {m1:+.4f} [{l1:+.4f}, {h1:+.4f}]   "
                  f"TF-{uni} {m2:+.4f} [{l2:+.4f}, {h2:+.4f}]")


if __name__ == "__main__":
    main()

"""Benchmarks whose clients are real institutions (v4).

heart     : UCI Heart Disease, 4 institutions (FLamby Fed-Heart-Disease split)
eicu_demo : open eICU-CRD demo, one client per hospital with >= min_site stays
eicu      : full eICU-CRD (credentialed; place patient.csv under data/eicu)

No DP is run here: with 4 to ~10 clients the client-level Gaussian mechanism
needs noise of the order of the whole aggregate, so a DP table on these
federations would only measure noise.
"""
from __future__ import annotations

import argparse

from .runner import run_single

TF = {"warmup": 5, "rho": 0.25}
STRATS = [("fedavg", {}), ("fedprox", {}), ("fedper", {}), ("fixed", dict(k_slow=4, k_med=4)),
          ("tierfed", TF), ("adacomm", {}), ("lag", {}), ("fedlama", {}), ("scaffold", {})]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["heart", "eicu_demo"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 52)))
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--min-site", type=int, default=80, help="eICU: minimum stays per hospital")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="results")
    a = ap.parse_args(argv)
    for ds in a.datasets:
        dkw = {"min_site": a.min_site} if ds.startswith("eicu") else {}
        for strat, skw in STRATS:
            for seed in a.seeds:
                run_single(dict(dataset=ds, dataset_kwargs=dkw, strategy=strat, strategy_kwargs=skw,
                                rounds=a.rounds, n_clients=0, seed=seed, device=a.device,
                                batch_size=32 if ds == "heart" else 64,
                                out_dir=a.out, tag="_real"), skip_existing=True)


if __name__ == "__main__":
    main()

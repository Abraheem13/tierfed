"""Main benchmark: every strategy x every seed on one dataset/model.

TierFed is part of the default strategy list (v2 omitted it and tested the
fixed schedule instead). Statistical tables are produced by
`src.analysis.collect`, which reads the logs; this script only runs.
"""
from __future__ import annotations

import argparse
import itertools
import json

from ..utils import get_logger
from .runner import run_single

log = get_logger()

STRATEGIES = ["fedavg", "fedprox", "scaffold", "fedper", "fedlama", "adacomm", "lag", "fixed", "tierfed"]
DEFAULT_KW = {"tierfed": {"warmup": 5, "rho": 0.25}, "fixed": {"k_slow": 5, "k_med": 2}}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Main federated benchmark")
    ap.add_argument("--dataset", default="diabetes")
    ap.add_argument("--dataset-kwargs", default=None, help="JSON dict")
    ap.add_argument("--model", default="mlp")
    ap.add_argument("--strategies", nargs="+", default=STRATEGIES)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 52)))
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--clients", type=int, default=20)
    ap.add_argument("--loss", default="weighted_ce")
    ap.add_argument("--rho", type=float, default=0.25)
    ap.add_argument("--membership-inference", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="results")
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)

    dkw = json.loads(args.dataset_kwargs) if args.dataset_kwargs else {}
    for strat, seed in itertools.product(args.strategies, args.seeds):
        skw = dict(DEFAULT_KW.get(strat, {}))
        if strat == "tierfed":
            skw["rho"] = args.rho
        run_single(dict(dataset=args.dataset, dataset_kwargs=dkw, model=args.model,
                        strategy=strat, strategy_kwargs=skw, rounds=args.rounds,
                        n_clients=args.clients, seed=seed, loss=args.loss,
                        membership_inference=args.membership_inference,
                        device=args.device, out_dir=args.out, tag=args.tag), skip_existing=True)


if __name__ == "__main__":
    main()

"""Scheduling x compression: FedPer and FedAvg under each compressor as
baselines, TierFed composed with each compressor."""
from __future__ import annotations

import argparse

from .runner import run_single

COMPRESSORS = [None, "quant8", "quant4", "topk", "stc", "sketch"]
TF = {"warmup": 5, "rho": 0.25}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Compression vs / with TierFed")
    ap.add_argument("--dataset", default="diabetes")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 52)))
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--clients", type=int, default=20)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)

    for strat in ("fedavg", "fedper", "tierfed"):
        for comp in COMPRESSORS:
            for seed in args.seeds:
                run_single(dict(dataset=args.dataset, strategy=strat, compressor=comp,
                                strategy_kwargs=TF if strat == "tierfed" else {},
                                rounds=args.rounds, n_clients=args.clients, seed=seed,
                                device=args.device, out_dir=args.out,
                                tag=f"_comp_{comp or 'none'}"), skip_existing=True)


if __name__ == "__main__":
    main()

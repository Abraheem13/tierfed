"""Ablations.

  budget        rho sweep for TierFed (+ solver = greedy vs exact)
  factorial     {private head} x {schedule}: FedAvg / FedPer / schedule-only / fixed
  periods       K_slow x K_med grid for the fixed schedule
  architecture  MLP / TabResNet / FT-Transformer with IDENTICAL depths for every
                strategy (taken from runner.ARCH), FedAvg / FedPer / fixed / TierFed
  loss          inverse-frequency / focal / LDAM / class-balanced, FedPer vs TierFed
  warmup        T_w sweep for TierFed
"""
from __future__ import annotations

import argparse
import itertools

from ..utils import get_logger
from .runner import run_single

log = get_logger()
TF = {"warmup": 5, "rho": 0.25}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Ablations")
    ap.add_argument("--kind", default="budget",
                    choices=["budget", "factorial", "periods", "architecture", "loss", "warmup", "all"])
    ap.add_argument("--dataset", default="diabetes")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 52)))
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--clients", type=int, default=20)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)
    kinds = ["budget", "factorial", "periods", "architecture", "loss", "warmup"] if args.kind == "all" else [args.kind]

    for kind in kinds:
        if kind == "budget":
            cfgs = [(f"rho{r}", dict(strategy="tierfed", strategy_kwargs={**TF, "rho": r}))
                    for r in (0.15, 0.25, 0.4, 0.6, 0.8)]
            cfgs += [(f"rho{r}_greedy", dict(strategy="tierfed", strategy_kwargs={**TF, "rho": r, "solver": "greedy"}))
                     for r in (0.25, 0.4)]
        elif kind == "factorial":
            cfgs = [("fedavg", dict(strategy="fedavg")),
                    ("fedper", dict(strategy="fedper")),
                    ("sched_only", dict(strategy="fixed", strategy_kwargs=dict(private_head=False, schedule=True))),
                    ("fixed", dict(strategy="fixed", strategy_kwargs=dict(private_head=True, schedule=True))),
                    ("tierfed_nohead", dict(strategy="tierfed", strategy_kwargs={**TF, "private_head": False})),
                    ("tierfed", dict(strategy="tierfed", strategy_kwargs=TF))]
        elif kind == "periods":
            cfgs = [(f"ks{ks}_km{km}", dict(strategy="fixed", strategy_kwargs=dict(k_slow=ks, k_med=km)))
                    for ks, km in itertools.product([2, 4, 8], [1, 2, 4, 8]) if km <= ks]
        elif kind == "architecture":
            cfgs = []
            for m in ["mlp", "resnet", "transformer"]:
                for s in ["fedavg", "fedper", "fixed", "tierfed"]:
                    cfgs.append((f"{m}_{s}", dict(model=m, strategy=s,
                                                  strategy_kwargs=TF if s == "tierfed" else {})))
        elif kind == "loss":
            cfgs = [(f"{l}_{s}", dict(loss=l, strategy=s, strategy_kwargs=TF if s == "tierfed" else {}))
                    for l in ["weighted_ce", "focal", "ldam", "cb"] for s in ["fedper", "tierfed"]]
        else:  # warmup
            cfgs = [(f"warm{w}", dict(strategy="tierfed", strategy_kwargs={**TF, "warmup": w}))
                    for w in (0, 2, 5, 10, 20)]
        for label, cfg in cfgs:
            for seed in args.seeds:
                run_single({**cfg, "seed": seed, "device": args.device, "out_dir": args.out,
                            "rounds": args.rounds, "n_clients": args.clients, "dataset": args.dataset,
                            "tag": f"_{kind}_{label}"}, skip_existing=True)


if __name__ == "__main__":
    main()

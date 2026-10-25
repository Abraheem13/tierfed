"""Privacy and attack-surface evaluation.

  1. Utility under client-level DP (per-client clipping, server noise) for
     FedAvg, FedPer, fixed and TierFed across noise multipliers, with the
     accountant charging only rounds in which something was transmitted.
  2. Update-inversion attack run INSIDE training on every strategy's real
     transmitted slice every `attack_every` rounds (post-warmup rounds included).
  3. Membership inference against every client's deployed model.
"""
from __future__ import annotations

import argparse

from .runner import run_single

TF = {"warmup": 5, "rho": 0.25}


def main(argv=None):
    ap = argparse.ArgumentParser(description="DP utility + inversion + membership inference")
    ap.add_argument("--dataset", default="diabetes")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 52)))
    ap.add_argument("--sigmas", nargs="+", type=float, default=[0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--strategies", nargs="+", default=["fedavg", "fedper", "fixed", "tierfed"])
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--clients", type=int, default=20)
    ap.add_argument("--attack-every", type=int, default=1)
    ap.add_argument("--attack-steps", type=int, default=100)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)

    for strat in args.strategies:
        skw = TF if strat == "tierfed" else {}
        # attack surface + membership inference, no DP
        for seed in args.seeds:
            run_single(dict(dataset=args.dataset, strategy=strat, strategy_kwargs=skw,
                            rounds=args.rounds, n_clients=args.clients, seed=seed,
                            attack_every=args.attack_every, attack_steps=args.attack_steps, membership_inference=True,
                            device=args.device, out_dir=args.out, tag="_attack"), skip_existing=True)
        for sigma in args.sigmas:
            for seed in args.seeds:
                dp = dict(enabled=True, sigma=sigma, max_norm=args.clip, delta=1e-5)
                run_single(dict(dataset=args.dataset, strategy=strat, strategy_kwargs=skw, dp=dp,
                                rounds=args.rounds, n_clients=args.clients, seed=seed,
                                device=args.device, out_dir=args.out, tag=f"_dp{sigma}"), skip_existing=True)


if __name__ == "__main__":
    main()

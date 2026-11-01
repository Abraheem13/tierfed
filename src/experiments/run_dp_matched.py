"""Matched-privacy comparison (v4/v5).

Every method is run at the SAME (epsilon, delta): its noise multiplier is
calibrated to its own number of planned releases with the exact (subsampled)
Gaussian RDP accountant, instead of sharing one sigma. Only data-independent
schedules are eligible under DP (see server.py): FedAvg, FedPer, fixed periods
and TierFed-DP. Client sampling is Poisson with rate q.

v5 adds the federation size N as a parameter (client-level noise scales with
1/(qN), so 20 sites is the hardest regime), the sparsest fixed period that
meets the quota, and the scale-free planner objective.
"""
from __future__ import annotations

import argparse

from .runner import run_single

FIXED4 = dict(k_slow=4, k_med=4, warmup=5)
FIXED8 = dict(k_slow=8, k_med=8, warmup=5)


def methods(eps, q, rounds, lam=1.0):
    return [("fedavg", {}, ""), ("fedper", {}, ""), ("fixed", FIXED4, ""), ("fixed", FIXED8, "_k8"),
            ("tierfed_dp", dict(rounds=rounds, warmup=5, epsilon=eps, q=q, lam=lam, rho=1.0), "")]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="diabetes")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 52)))
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--clients", type=int, default=20)
    ap.add_argument("--partition", default=None, help="dataset partition override (100 sites: dirichlet)")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--grid", nargs="+", default=["0.5:1", "0.5:2", "0.5:4", "0.5:8", "1.0:4"],
                    help="q:epsilon pairs")
    ap.add_argument("--lams", nargs="+", type=float, default=[0.3, 3.0, 10.0],
                    help="extra TierFed-DP lambda values (lambda=1 is in the grid)")
    ap.add_argument("--lam-at", default="1.0:4", help="q:epsilon at which the lambda sweep runs")
    ap.add_argument("--only", nargs="+", default=None, help="restrict to these strategies")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="results")
    a = ap.parse_args(argv)

    dkw = {"partition": a.partition, "alpha": a.alpha} if a.partition else None
    jobs = []
    for qe in a.grid:
        q, eps = (float(v) for v in qe.split(":"))
        for strat, skw, suffix in methods(eps, q, a.rounds):
            jobs.append((strat, skw, q, eps, f"_dpe{eps:g}_q{q:g}{suffix}"))
    q0, e0 = (float(v) for v in a.lam_at.split(":"))
    for lam in a.lams:
        jobs.append(("tierfed_dp", dict(rounds=a.rounds, warmup=5, epsilon=e0, q=q0, lam=lam, rho=1.0),
                     q0, e0, f"_dpe{e0:g}_q{q0:g}_lam{lam:g}"))
    for strat, skw, q, eps, tag in jobs:
        if a.only and strat not in a.only:
            continue
        for seed in a.seeds:
            dp = dict(enabled=True, sigma=1.0, max_norm=a.clip, delta=1e-5, sample_rate=q, target_epsilon=eps)
            run_single(dict(dataset=a.dataset, dataset_kwargs=dkw, strategy=strat, strategy_kwargs=skw, dp=dp,
                            rounds=a.rounds, n_clients=a.clients, seed=seed, device=a.device,
                            out_dir=a.out, tag=tag), skip_existing=True)


if __name__ == "__main__":
    main()

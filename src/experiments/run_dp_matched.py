"""Matched-privacy comparison (v4).

Every method is run at the SAME (epsilon, delta): its noise multiplier is
calibrated to its own number of planned releases with the exact (subsampled)
Gaussian RDP accountant, instead of sharing one sigma. Only data-independent
schedules are eligible under DP (see server.py): FedAvg, FedPer, a fixed period
and TierFed-DP. Client sampling is Poisson with rate q.
"""
from __future__ import annotations

import argparse

from .runner import run_single

FIXED = dict(k_slow=4, k_med=4, warmup=5)


def methods(eps, q, rounds, lam=1.0):
    return [("fedavg", {}), ("fedper", {}), ("fixed", FIXED),
            ("tierfed_dp", dict(rounds=rounds, warmup=5, epsilon=eps, q=q, lam=lam, rho=1.0))]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="diabetes")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 52)))
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--clients", type=int, default=20)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--grid", nargs="+", default=["0.5:1", "0.5:2", "0.5:4", "0.5:8", "1.0:4"],
                    help="q:epsilon pairs")
    ap.add_argument("--lams", nargs="+", type=float, default=[0.1, 10.0],
                    help="extra TierFed-DP lambda values at q=1, eps=4 (lambda=1 is in the grid)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="results")
    a = ap.parse_args(argv)

    jobs = []
    for qe in a.grid:
        q, eps = (float(v) for v in qe.split(":"))
        for strat, skw in methods(eps, q, a.rounds):
            jobs.append((strat, skw, q, eps, f"_dpe{eps:g}_q{q:g}"))
    for lam in a.lams:
        jobs.append(("tierfed_dp", dict(rounds=a.rounds, warmup=5, epsilon=4.0, q=1.0, lam=lam, rho=1.0),
                     1.0, 4.0, f"_dpe4_q1_lam{lam:g}"))
    for strat, skw, q, eps, tag in jobs:
        for seed in a.seeds:
            dp = dict(enabled=True, sigma=1.0, max_norm=a.clip, delta=1e-5, sample_rate=q, target_epsilon=eps)
            run_single(dict(dataset=a.dataset, strategy=strat, strategy_kwargs=skw, dp=dp,
                            rounds=a.rounds, n_clients=a.clients, seed=seed, device=a.device,
                            out_dir=a.out, tag=tag), skip_existing=True)


if __name__ == "__main__":
    main()

"""Scalability and overhead: wall-clock per round, controller time, peak memory
and accuracy as the client count grows. The specialty partition cannot yield
100 clients (there are ~73 specialties), so scaling uses the Dirichlet
partition with the same alpha for every method. TierFed is included (v2's
table had no TierFed runs)."""
from __future__ import annotations

import argparse
import time

import torch

from .runner import run_single

TF = {"warmup": 5, "rho": 0.25}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Scaling / overhead")
    ap.add_argument("--dataset", default="diabetes")
    ap.add_argument("--client-counts", nargs="+", type=int, default=[10, 20, 50, 100])
    ap.add_argument("--strategies", nargs="+", default=["fedavg", "fedper", "fedlama", "tierfed"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)

    for n in args.client_counts:
        for strat in args.strategies:
            for seed in args.seeds:
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
                res = run_single(dict(dataset=args.dataset,
                                      dataset_kwargs={"partition": "dirichlet", "alpha": args.alpha},
                                      strategy=strat, strategy_kwargs=TF if strat == "tierfed" else {},
                                      rounds=args.rounds, n_clients=n, seed=seed, device=args.device,
                                      out_dir=args.out, tag=f"_scale{n}"), skip_existing=True)
                el = time.perf_counter() - t0
                if not res:
                    continue
                mem = torch.cuda.max_memory_allocated() / 2 ** 20 if torch.cuda.is_available() else float("nan")
                # append timing to the summary file so collect.py can read it
                from pathlib import Path
                import json
                p = Path(args.out) / "logs"
                tag = f"{args.dataset}_mlp_{strat}_R{args.rounds}_C{res['dataset']['n_clients']}_s{seed}_scale{n}"
                f = p / f"{tag}_summary.json"
                if f.exists():
                    d = json.loads(f.read_text())
                    d["summary"].update({"wall_s": el, "s_per_round": el / args.rounds, "peak_mem_mb": mem})
                    f.write_text(json.dumps(d, indent=2))


if __name__ == "__main__":
    main()

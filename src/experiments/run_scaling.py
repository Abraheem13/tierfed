"""Scalability and overhead study.

Answers "the article fails to discuss computational overhead and scalability":
wall-clock per round, peak memory, and accuracy as the client count grows.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import torch

from ..utils import get_logger
from .runner import run_single

log = get_logger()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Scaling / overhead")
    ap.add_argument("--dataset", default="diabetes")
    ap.add_argument("--client-counts", nargs="+", type=int, default=[10, 20, 50, 100])
    ap.add_argument("--strategies", nargs="+", default=["fedavg", "fedper", "fedlama", "nested"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)

    rows = []
    for n in args.client_counts:
        for strat in args.strategies:
            for seed in args.seeds:
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
                res = run_single(dict(dataset=args.dataset, strategy=strat, rounds=args.rounds,
                                      n_clients=n, seed=seed, device=args.device,
                                      out_dir=args.out, tag=f"_scale{n}"))
                el = time.perf_counter() - t0
                mem = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else float("nan")
                rows.append({"n_clients": n, "strategy": strat, "seed": seed,
                             "wall_s": el, "s_per_round": el / args.rounds,
                             "peak_mem_mb": mem, **res["summary"]})
    df = pd.DataFrame(rows)
    p = Path(args.out) / "tables"; p.mkdir(parents=True, exist_ok=True)
    df.to_csv(p / "scaling_raw.csv", index=False)
    g = df.groupby(["n_clients", "strategy"]).agg(
        wall_s=("wall_s", "mean"), s_per_round=("s_per_round", "mean"),
        peak_mem_mb=("peak_mem_mb", "mean"),
        auroc=("fed_auroc_final", "mean"), upload_mb=("total_upload_mb", "mean")).reset_index()
    g.to_csv(p / "scaling_summary.csv", index=False)
    log.info("\n" + g.to_string(index=False))
    return df


if __name__ == "__main__":
    main()

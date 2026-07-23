"""Single (strategy, seed) run -- the unit of parallelism.

Tabular federated runs are dominated by Python/dataloader overhead rather than
matrix multiplication, so launching many of these in parallel across CPU cores
scales far better than making one run use more GPU.
"""
from __future__ import annotations

import argparse
import json

from ..utils import get_logger
from .runner import run_single

log = get_logger()


def main(argv=None):
    ap = argparse.ArgumentParser(description="One federated run")
    ap.add_argument("--dataset", default="diabetes")
    ap.add_argument("--dataset-kwargs", default=None)
    ap.add_argument("--model", default="mlp")
    ap.add_argument("--model-kwargs", default=None)
    ap.add_argument("--strategy", default="nested")
    ap.add_argument("--strategy-kwargs", default=None)
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--clients", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--loss", default="weighted_ce")
    ap.add_argument("--compressor", default=None)
    ap.add_argument("--dp-sigma", type=float, default=0.0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int, default=1,
                    help="torch intra-op threads; keep at 1 when running many jobs in parallel")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="results")
    a = ap.parse_args(argv)

    import torch
    torch.set_num_threads(max(1, a.threads))

    J = lambda s: json.loads(s) if s else {}   # noqa: E731
    cfg = dict(dataset=a.dataset, dataset_kwargs=J(a.dataset_kwargs),
               model=a.model, model_kwargs=J(a.model_kwargs),
               strategy=a.strategy, strategy_kwargs=J(a.strategy_kwargs),
               rounds=a.rounds, n_clients=a.clients, seed=a.seed, loss=a.loss,
               compressor=a.compressor, device=a.device, out_dir=a.out, tag=a.tag)
    if a.dp_sigma > 0:
        cfg["dp"] = dict(enabled=True, sigma=a.dp_sigma, max_norm=1.0, delta=1e-5)
    run_single(cfg)


if __name__ == "__main__":
    main()

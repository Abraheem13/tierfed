"""Single (strategy, seed) run: the unit of parallelism and of resumption."""
from __future__ import annotations

import argparse
import json

from .runner import run_single


def main(argv=None):
    ap = argparse.ArgumentParser(description="One federated run")
    ap.add_argument("--dataset", default="diabetes")
    ap.add_argument("--dataset-kwargs", default=None)
    ap.add_argument("--model", default="mlp")
    ap.add_argument("--model-kwargs", default=None)
    ap.add_argument("--strategy", default="tierfed")
    ap.add_argument("--strategy-kwargs", default=None)
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--clients", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--loss", default="weighted_ce")
    ap.add_argument("--optimizer", default="adam")
    ap.add_argument("--compressor", default=None)
    ap.add_argument("--dp-sigma", type=float, default=0.0)
    ap.add_argument("--dp-clip", type=float, default=1.0)
    ap.add_argument("--attack-every", type=int, default=0)
    ap.add_argument("--membership-inference", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="results")
    ap.add_argument("--skip-existing", action="store_true")
    a = ap.parse_args(argv)

    J = lambda s: json.loads(s) if s else {}   # noqa: E731
    cfg = dict(dataset=a.dataset, dataset_kwargs=J(a.dataset_kwargs),
               model=a.model, model_kwargs=J(a.model_kwargs),
               strategy=a.strategy, strategy_kwargs=J(a.strategy_kwargs),
               rounds=a.rounds, n_clients=a.clients, seed=a.seed, loss=a.loss, optimizer=a.optimizer,
               compressor=a.compressor, device=a.device, out_dir=a.out, tag=a.tag,
               attack_every=a.attack_every, membership_inference=a.membership_inference)
    if a.dp_sigma > 0:
        cfg["dp"] = dict(enabled=True, sigma=a.dp_sigma, max_norm=a.dp_clip, delta=1e-5)
    run_single(cfg, skip_existing=a.skip_existing)


if __name__ == "__main__":
    main()

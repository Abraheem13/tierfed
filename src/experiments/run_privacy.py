"""Privacy evaluation.

Two parts, both previously absent:
  1. Utility under client-level differential privacy across noise multipliers,
     with the (epsilon, delta) budget reported per configuration.
  2. A gradient-inversion attack run against the ACTUAL transmitted slice of
     each strategy, so privacy is measured rather than asserted.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from ..analysis.stats import summarise_runs
from ..data import load_dataset
from ..models import assign_tiers, build_model, tier_keys
from ..privacy import DPConfig, evaluate_inversion
from ..strategies import build_strategy
from ..utils import clone_state, get_device, get_logger, set_seed
from .runner import run_single

log = get_logger()
METRICS = ["fed_auroc_final", "fed_auprc_final", "fed_f1_final", "total_upload_mb"]


def inversion_study(dataset="diabetes", n_clients=20, seed=42, device="auto",
                    steps=300, batch=8, out="results"):
    """Attack each strategy's transmitted update; fewer exposed tensors should
    translate into a measurably worse reconstruction."""
    set_seed(seed)
    dev = get_device(device)
    ds = load_dataset(dataset, n_clients=n_clients, seed=seed)
    model = build_model("mlp", ds.input_shape).to(dev)
    tiers = assign_tiers(model)
    names = [n for n, _ in model.named_parameters()]
    sizes = {n: p.numel() for n, p in model.named_parameters()}
    head = tier_keys(tiers, "fast")

    idx = ds.client_indices[0][:batch]
    x = torch.as_tensor(ds.x_train[idx], dtype=torch.float32).to(dev)
    y = torch.as_tensor(ds.y_train[idx], dtype=torch.long).to(dev)

    base = clone_state(model.state_dict())
    loss = torch.nn.CrossEntropyLoss()(model(x), y)
    grads = torch.autograd.grad(loss, [p for _, p in model.named_parameters()])
    full = {n: g.detach() for (n, _), g in zip(model.named_parameters(), grads)}

    rows = []
    for sname in ("fedavg", "fedper", "nested"):
        st = build_strategy(sname, names, sizes, tiers=tiers, head_keys=head)
        keys = st.send_keys(2)                      # a representative non-full round
        upd = {k: full[k] for k in keys if k in full}
        model.load_state_dict(base)
        r = evaluate_inversion(model, upd, x, dev, steps=steps, seed=seed)
        rows.append({"strategy": sname, "round": 2, **r})
        log.info(f"{sname:8s} exposed {r['n_params_exposed']:.0f} params "
                 f"| NMSE {r['inversion_nmse']:.4f} | corr {r['inversion_feature_corr']:.4f}")
    df = pd.DataFrame(rows)
    p = Path(out) / "tables"; p.mkdir(parents=True, exist_ok=True)
    df.to_csv(p / "privacy_inversion.csv", index=False)
    return df


def main(argv=None):
    ap = argparse.ArgumentParser(description="DP utility + inversion attack")
    ap.add_argument("--dataset", default="diabetes")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 47)))
    ap.add_argument("--sigmas", nargs="+", type=float, default=[0.0, 0.5, 1.0, 2.0])
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--clients", type=int, default=20)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="results")
    ap.add_argument("--skip-inversion", action="store_true")
    args = ap.parse_args(argv)

    rows = []
    for strat in ("fedavg", "nested"):
        for sigma in args.sigmas:
            for seed in args.seeds:
                dp = dict(enabled=sigma > 0, sigma=sigma, max_norm=1.0, delta=1e-5)
                res = run_single(dict(dataset=args.dataset, strategy=strat, dp=dp,
                                      rounds=args.rounds, n_clients=args.clients, seed=seed,
                                      device=args.device, out_dir=args.out, tag=f"_dp{sigma}"))
                rows.append({"strategy": strat, "sigma": sigma, "seed": seed,
                             "epsilon": res["dp"]["dp_epsilon"], **res["summary"]})
    df = pd.DataFrame(rows)
    p = Path(args.out) / "tables"; p.mkdir(parents=True, exist_ok=True)
    df.to_csv(p / "privacy_dp_raw.csv", index=False)
    s = summarise_runs(df, ["strategy", "sigma"], [m for m in METRICS if m in df])
    s.to_csv(p / "privacy_dp_summary.csv", index=False)
    log.info("\n" + s.to_string(index=False))
    if not args.skip_inversion:
        inversion_study(args.dataset, args.clients, args.seeds[0], args.device, out=args.out)
    return df


if __name__ == "__main__":
    main()

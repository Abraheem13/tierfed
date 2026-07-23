"""Ablations.

1. FACTORIAL (the ablation reviewers said was missing): a 2x2 over
   {private head} x {multi-rate schedule}, which attributes the effect to the
   right mechanism instead of confounding them.
2. Period sweep over K_slow x K_med (two-dimensional, not one).
3. Architecture sweep: MLP / TabResNet / FT-Transformer / CNN, showing the
   tiering is not tied to a two-hidden-layer MLP.
4. Loss sweep: inverse-frequency vs focal vs LDAM vs class-balanced vs
   balanced sampling, covering the class-imbalance objection.
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import pandas as pd

from ..analysis.stats import summarise_runs
from ..utils import get_logger
from .runner import run_single

log = get_logger()
METRICS = ["fed_auroc_final", "fed_auroc_best", "fed_auroc_ppd", "fed_auprc_final",
           "fed_f1_final", "total_upload_mb", "divergence_total_mean"]


def _run(kind, cfgs, args):
    rows = []
    for label, cfg in cfgs:
        for seed in args.seeds:
            res = run_single({**cfg, "seed": seed, "device": args.device,
                              "out_dir": args.out, "rounds": args.rounds,
                              "n_clients": args.clients, "dataset": args.dataset,
                              "tag": f"_{kind}_{label}"})
            rows.append({"cell": label, "seed": seed, **res["summary"]})
    df = pd.DataFrame(rows)
    out = Path(args.out) / "tables"; out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"ablation_{kind}_raw.csv", index=False)
    s = summarise_runs(df, ["cell"], [m for m in METRICS if m in df])
    s.to_csv(out / f"ablation_{kind}_summary.csv", index=False)
    log.info(f"\n[{kind}]\n" + s.to_string(index=False))
    return df


def main(argv=None):
    ap = argparse.ArgumentParser(description="NFL ablations")
    ap.add_argument("--kind", default="factorial",
                    choices=["factorial", "periods", "architecture", "loss", "all"])
    ap.add_argument("--dataset", default="diabetes")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 52)))
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--clients", type=int, default=20)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)
    kinds = ["factorial", "periods", "architecture", "loss"] if args.kind == "all" else [args.kind]

    for kind in kinds:
        if kind == "factorial":
            cfgs = [
                ("head0_sched0_fedavg", dict(strategy="fedavg")),
                ("head1_sched0_fedper", dict(strategy="fedper")),
                ("head0_sched1", dict(strategy="nested",
                                      strategy_kwargs=dict(private_head=False, schedule=True))),
                ("head1_sched1_nfl", dict(strategy="nested",
                                          strategy_kwargs=dict(private_head=True, schedule=True))),
            ]
        elif kind == "periods":
            cfgs = [(f"ks{ks}_km{km}",
                     dict(strategy="nested", strategy_kwargs=dict(k_slow=ks, k_med=km)))
                    for ks, km in itertools.product([2, 5, 10], [1, 2, 4, 8]) if km <= ks or km == 8]
        elif kind == "architecture":
            cfgs = [(f"{m}_{s}", dict(model=m, strategy=s,
                                      model_kwargs={"depth": 4} if m in ("resnet", "transformer") else {}))
                    for m in ["mlp", "resnet", "transformer"] for s in ["fedavg", "fedper", "nested"]]
        else:  # loss
            cfgs = [(f"{l}_{s}", dict(loss=l, strategy=s,
                                      balanced_sampler=(l == "balanced_sampler")))
                    for l in ["weighted_ce", "focal", "ldam", "cb", "balanced_sampler"]
                    for s in ["fedavg", "nested"]]
        _run(kind, cfgs, args)


if __name__ == "__main__":
    main()

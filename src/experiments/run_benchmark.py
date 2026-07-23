"""Main benchmark: every strategy x every seed x every dataset.

Includes FedPer and FedLAMA, the baselines whose absence was called
disqualifying, and defaults to 10 seeds.
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import pandas as pd

from ..analysis.stats import compare_all, non_inferiority, summarise_runs
from ..utils import get_logger, save_json
from .runner import run_single

log = get_logger()

STRATEGIES = ["fedavg", "fedprox", "fedper", "fedlama", "scaffold", "nested"]
METRICS = ["fed_auroc_final", "fed_auroc_best", "fed_auroc_ppd", "fed_auroc_earlystop_value",
           "fed_auprc_final", "fed_auprc_best", "fed_f1_final", "fed_f1_best_final",
           "fed_ece_final", "global_auroc_final", "total_upload_mb",
           "divergence_total_mean"]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Main federated benchmark")
    ap.add_argument("--dataset", default="diabetes")
    ap.add_argument("--dataset-kwargs", default=None, help="JSON dict")
    ap.add_argument("--model", default="mlp")
    ap.add_argument("--strategies", nargs="+", default=STRATEGIES)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 52)))
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--clients", type=int, default=20)
    ap.add_argument("--partition", default=None)
    ap.add_argument("--loss", default="weighted_ce")
    ap.add_argument("--k-slow", type=int, default=5)
    ap.add_argument("--k-med", type=int, default=2)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)

    import json
    dkw = json.loads(args.dataset_kwargs) if args.dataset_kwargs else {}
    if args.partition:
        dkw["partition"] = args.partition

    rows = []
    for strat, seed in itertools.product(args.strategies, args.seeds):
        skw = {"k_slow": args.k_slow, "k_med": args.k_med} if strat.startswith("nested") else {}
        res = run_single(dict(dataset=args.dataset, dataset_kwargs=dkw, model=args.model,
                              strategy=strat, strategy_kwargs=skw, rounds=args.rounds,
                              n_clients=args.clients, seed=seed, loss=args.loss,
                              device=args.device, out_dir=args.out))
        rows.append({"strategy": strat, "seed": seed, "dataset": args.dataset,
                     "model": args.model, **res["summary"]})

    df = pd.DataFrame(rows)
    out = Path(args.out); (out / "tables").mkdir(parents=True, exist_ok=True)
    tag = f"{args.dataset}_{args.model}"
    df.to_csv(out / "tables" / f"benchmark_{tag}_raw.csv", index=False)

    summ = summarise_runs(df, ["strategy"], [m for m in METRICS if m in df])
    summ.to_csv(out / "tables" / f"benchmark_{tag}_summary.csv", index=False)

    tests = []
    for base in ("fedavg", "fedprox", "fedper", "fedlama"):
        if base in df["strategy"].values:
            t = compare_all(df[df.strategy.isin(["nested", base])], "strategy", base,
                            [m for m in METRICS if m in df])
            tests.append(t)
    if tests:
        allt = pd.concat(tests, ignore_index=True)
        allt.to_csv(out / "tables" / f"benchmark_{tag}_tests.csv", index=False)

    # Peak AUROC is an equivalence claim: test non-inferiority, never "n.s.".
    ni = {}
    for base in ("fedavg", "fedprox", "fedper", "fedlama"):
        if base in df["strategy"].values:
            a = df[df.strategy == "nested"].sort_values("seed")["fed_auroc_best"].values
            b = df[df.strategy == base].sort_values("seed")["fed_auroc_best"].values
            n = min(len(a), len(b))
            ni[f"nested_vs_{base}"] = non_inferiority(a[:n], b[:n], margin=0.01)
    save_json(out / "tables" / f"benchmark_{tag}_noninferiority.json", ni)

    log.info("\n" + summ.to_string(index=False))
    return df


if __name__ == "__main__":
    main()

"""Compression benchmark.

Answers: "the 32.6% bandwidth saving is never benchmarked against standard
compression". Runs FedAvg under each compressor as a baseline, then NFL alone,
then NFL COMPOSED with each compressor -- establishing that the schedule is
complementary to, not competing with, quantisation/sparsification/sketching.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..analysis.stats import summarise_runs
from ..utils import get_logger
from .runner import run_single

log = get_logger()
COMPRESSORS = [None, "quant8", "quant4", "topk", "stc", "sketch"]
METRICS = ["fed_auroc_final", "fed_auroc_best", "fed_auroc_ppd", "fed_auprc_final",
           "fed_f1_final", "total_upload_mb"]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Compression vs / with NFL")
    ap.add_argument("--dataset", default="diabetes")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 47)))
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--clients", type=int, default=20)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)

    rows = []
    for strat in ("fedavg", "nested"):
        for comp in COMPRESSORS:
            for seed in args.seeds:
                res = run_single(dict(dataset=args.dataset, strategy=strat, compressor=comp,
                                      rounds=args.rounds, n_clients=args.clients, seed=seed,
                                      device=args.device, out_dir=args.out,
                                      tag=f"_comp_{comp or 'none'}"))
                rows.append({"strategy": strat, "compressor": comp or "none",
                             "seed": seed, **res["summary"]})
    df = pd.DataFrame(rows)
    out = Path(args.out) / "tables"; out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "compression_raw.csv", index=False)
    s = summarise_runs(df, ["strategy", "compressor"], [m for m in METRICS if m in df])
    # Bandwidth reduction relative to uncompressed FedAvg, the honest reference.
    ref = s[(s.strategy == "fedavg") & (s.compressor == "none")]["total_upload_mb_mean"]
    if len(ref):
        s["upload_reduction_pct"] = 100 * (1 - s["total_upload_mb_mean"] / float(ref.iloc[0]))
    s.to_csv(out / "compression_summary.csv", index=False)
    log.info("\n" + s.to_string(index=False))
    return df


if __name__ == "__main__":
    main()

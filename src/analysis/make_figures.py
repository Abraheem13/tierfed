"""Regenerate every figure of the paper from results/. Figures that have no
runs are skipped with a message (the manuscript uses \\IfFileExists)."""
from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .figures import (bandwidth_frontier, divergence_figure, factorial_figure, privacy_tradeoff,
                      schedule_figure, trajectory_figure)
from .collect import load_summaries, NAMES


def _hist(pattern):
    runs = []
    for f in sorted(glob.glob(pattern)):
        runs.append(pd.read_csv(f).sort_values("round"))
    return runs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="paper/figures")
    ap.add_argument("--dataset", default="diabetes")
    ap.add_argument("--model", default="mlp")
    a = ap.parse_args(argv)
    R = Path(a.results); O = Path(a.out); O.mkdir(parents=True, exist_ok=True)
    L = str(R / "logs")
    key = f"{a.dataset}_{a.model}"

    hist = {}
    for s in ["fedper", "fedavg", "fedlama", "adacomm", "lag", "fixed", "tierfed"]:
        runs = _hist(f"{L}/{key}_{s}_R*_C*_s*_history.csv")
        runs = [r for r, f in zip(runs, sorted(glob.glob(f"{L}/{key}_{s}_R*_C*_s*_history.csv")))
                if re.search(r"_s\d+_history", f)]
        if runs:
            hist[NAMES[s]] = runs
    if hist:
        trajectory_figure(hist, out=O / "fig_traj.pdf")
        trajectory_figure(hist, metric="upload_mb", ylabel="Cumulative upload (MB)", out=O / "fig_upload.pdf")
        if "TierFed" in hist:
            divergence_figure({"TierFed": hist["TierFed"]}, out=O / "fig_divergence.pdf")
            schedule_figure(hist["TierFed"], out=O / "fig_schedule.pdf")
    else:
        print("no main-benchmark histories: trajectory figures skipped")

    df = load_summaries(R / "logs")
    if df.empty:
        return
    tag = df["tag"].fillna("")
    main_df = df[(df.dataset == a.dataset) & (df.model == a.model) & (tag == "")]
    comp = df[tag.str.startswith("_comp_")]
    budget = df[tag.str.startswith("_budget_")]
    fr = pd.concat([main_df, comp, budget])
    if not fr.empty:
        g = fr.groupby(["strategy", "compressor", "tag"]).agg(
            total_upload_mb_mean=("total_upload_mb", "mean"),
            fed_auroc_final_mean=("fed_auroc_final", "mean")).reset_index()
        bandwidth_frontier(g, group=("strategy", "compressor"), out=O / "fig_frontier.pdf")
    fac = df[tag.str.startswith("_factorial_")]
    if not fac.empty:
        s = fac.groupby("tag").agg(fed_auroc_final_mean=("fed_auroc_final", "mean"),
                                   fed_auroc_final_std=("fed_auroc_final", "std")).reset_index()
        s["cell"] = s["tag"].str.replace("_factorial_", "")
        factorial_figure(s, out=O / "fig_factorial.pdf")
    dp = df[tag.str.startswith("_dp")]
    if not dp.empty:
        s = dp.groupby(["strategy", "dp_sigma"]).agg(fed_auroc_final_mean=("fed_auroc_final", "mean"),
                                                     fed_auroc_final_std=("fed_auroc_final", "std")).reset_index()
        s = s.rename(columns={"dp_sigma": "sigma"})
        privacy_tradeoff(s, out=O / "fig_privacy.pdf")
    print("figures written to", O)


if __name__ == "__main__":
    main()

"""Regenerate every paper figure from results/ artefacts."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .figures import (bandwidth_frontier, divergence_figure, factorial_figure,
                      forest_plot, load_histories, privacy_tradeoff, trajectory_figure)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--dataset", default="diabetes")
    ap.add_argument("--model", default="mlp")
    a = ap.parse_args(argv)
    R = Path(a.results); F = R / "figures"; F.mkdir(parents=True, exist_ok=True)
    made = []

    hists = load_histories(R / "logs", f"{a.dataset}_{a.model}_*_history.csv")
    hists = {k: v for k, v in hists.items() if v}
    if hists:
        made.append(trajectory_figure(hists, "fed_auroc", "Federated AUROC", F / "fig_auroc.png"))
        made.append(trajectory_figure(hists, "fed_auprc", "Federated AUPRC", F / "fig_auprc.png"))
        made.append(trajectory_figure(hists, "fed_f1", "Federated F1 (positive class)", F / "fig_f1.png"))
        if any("divergence_slow" in v[0].columns for v in hists.values()):
            made.append(divergence_figure(hists, F / "fig_divergence.png"))

    T = R / "tables"
    fac = T / "ablation_factorial_summary.csv"
    if fac.exists():
        made.append(factorial_figure(pd.read_csv(fac), out=F / "fig_factorial.png"))
    comp = T / "compression_summary.csv"
    if comp.exists():
        made.append(bandwidth_frontier(pd.read_csv(comp), out=F / "fig_frontier.png"))
    dp = T / "privacy_dp_summary.csv"
    if dp.exists():
        made.append(privacy_tradeoff(pd.read_csv(dp), out=F / "fig_privacy.png"))
    tests = T / f"benchmark_{a.dataset}_{a.model}_tests.csv"
    if tests.exists():
        made.append(forest_plot(pd.read_csv(tests), out=F / "fig_forest.png"))

    print("figures written:")
    for m in made:
        if m:
            print(" ", m)


if __name__ == "__main__":
    main()

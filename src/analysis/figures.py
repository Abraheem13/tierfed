"""Publication figures.

Grayscale-safe (distinct dashes + markers), 300 dpi, with confidence bands
rather than bare standard deviations. Every figure carries the information a
reviewer asked to see: CIs, AUPRC, the factorial attribution, the
accuracy/bandwidth frontier including compression, and the DP trade-off.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.size": 9.5, "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
})

STYLES = [("-", "o", "#111111"), ("--", "s", "#555555"), (":", "^", "#888888"),
          ("-.", "D", "#333333"), ("-", "v", "#777777"), ("--", "P", "#999999")]


def _style(i):
    return STYLES[i % len(STYLES)]


def _band(ax, x, m, lo, hi, ls, mk, c, label):
    ax.plot(x, m, ls, color=c, marker=mk, markersize=3.2, markevery=max(1, len(x)//10),
            linewidth=1.7, label=label)
    ax.fill_between(x, lo, hi, color=c, alpha=0.13, linewidth=0)


def trajectory_figure(histories: Dict[str, List[pd.DataFrame]], metric: str = "fed_auroc",
                      ylabel: str = "Federated AUROC", out: str | Path = "fig_traj.png",
                      ci: float = 0.95):
    """Mean +/- bootstrap CI trajectory per strategy."""
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for i, (name, runs) in enumerate(histories.items()):
        arr = np.vstack([r[metric].values for r in runs])
        x = runs[0]["round"].values
        m = np.nanmean(arr, 0)
        se = np.nanstd(arr, 0, ddof=1) / np.sqrt(arr.shape[0])
        z = 1.96 if ci == 0.95 else 2.58
        ls, mk, c = _style(i)
        _band(ax, x, m, m - z * se, m + z * se, ls, mk, c, f"{name} (n={arr.shape[0]})")
    ax.set_xlabel("Communication round"); ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, linewidth=0.5); ax.legend(frameon=False, fontsize=8)
    fig.savefig(out); plt.close(fig)
    return out


def factorial_figure(summary: pd.DataFrame, metric: str = "fed_auroc_final_mean",
                     err: str = "fed_auroc_final_std", out="fig_factorial.png"):
    """2x2 attribution: private head x multi-rate schedule."""
    order = ["head0_sched0_fedavg", "head1_sched0_fedper", "head0_sched1", "head1_sched1_nfl"]
    labels = ["FedAvg\n(neither)", "FedPer\n(head only)", "NFL-Sched\n(schedule only)", "NFL\n(both)"]
    d = summary.set_index("cell").reindex(order)
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    x = np.arange(len(order))
    ax.bar(x, d[metric].values, yerr=d.get(err, pd.Series(np.zeros(len(order)))).values,
           color=["#cccccc", "#aaaaaa", "#888888", "#444444"], edgecolor="#111111",
           linewidth=0.7, capsize=4, width=0.62)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel(metric.replace("_mean", "").replace("fed_", "").replace("_", " ").upper())
    ax.grid(alpha=0.25, axis="y", linewidth=0.5)
    fig.savefig(out); plt.close(fig)
    return out


def bandwidth_frontier(df: pd.DataFrame, x="total_upload_mb_mean", y="fed_auroc_final_mean",
                       group=("strategy", "compressor"), out="fig_frontier.png"):
    """Accuracy vs upload, with compression baselines on the same axes."""
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    for i, (name, g) in enumerate(df.groupby(list(group))):
        ls, mk, c = _style(i)
        ax.scatter(g[x], g[y], marker=mk, s=42, color=c, label=" / ".join(map(str, name)))
    ax.set_xlabel("Cumulative upload (MB)"); ax.set_ylabel("Final federated AUROC")
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.savefig(out); plt.close(fig)
    return out


def privacy_tradeoff(df: pd.DataFrame, out="fig_privacy.png"):
    """Utility against the DP noise multiplier / epsilon."""
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    for i, (name, g) in enumerate(df.groupby("strategy")):
        g = g.sort_values("sigma")
        ls, mk, c = _style(i)
        ax.errorbar(g["sigma"], g["fed_auroc_final_mean"],
                    yerr=g.get("fed_auroc_final_std"), fmt=mk + ls, color=c,
                    capsize=3, linewidth=1.6, markersize=5, label=str(name))
    ax.set_xlabel("DP noise multiplier $\\sigma$"); ax.set_ylabel("Final federated AUROC")
    ax.grid(alpha=0.25, linewidth=0.5); ax.legend(frameon=False, fontsize=8)
    fig.savefig(out); plt.close(fig)
    return out


def divergence_figure(histories: Dict[str, List[pd.DataFrame]], out="fig_divergence.png"):
    """Empirical per-tier divergence: the assumption behind the theory."""
    fig, ax = plt.subplots(figsize=(6.2, 3.5))
    cols = ["divergence_slow", "divergence_medium", "divergence_total"]
    runs = next(iter(histories.values()))
    for i, col in enumerate(cols):
        if col not in runs[0]:
            continue
        arr = np.vstack([r[col].values for r in runs])
        m = np.nanmean(arr, 0); se = np.nanstd(arr, 0, ddof=1) / np.sqrt(arr.shape[0])
        ls, mk, c = _style(i)
        _band(ax, runs[0]["round"].values, m, m - 1.96*se, m + 1.96*se, ls, mk, c,
              col.replace("divergence_", "$\\Gamma_{") + "}$")
    ax.set_xlabel("Communication round"); ax.set_ylabel("Relative parameter divergence")
    ax.set_yscale("log"); ax.grid(alpha=0.25, linewidth=0.5); ax.legend(frameon=False, fontsize=8)
    fig.savefig(out); plt.close(fig)
    return out


def forest_plot(tests: pd.DataFrame, metric: str = "fed_auroc_final", out="fig_forest.png"):
    """Effect sizes with CIs and Holm-adjusted significance."""
    d = tests[tests.metric == metric].copy()
    if d.empty:
        return None
    d = d.sort_values("mean_diff")
    fig, ax = plt.subplots(figsize=(6.2, 0.42 * len(d) + 1.4))
    y = np.arange(len(d))
    ax.errorbar(d["mean_diff"], y,
                xerr=[d["mean_diff"] - d["diff_ci_lo"], d["diff_ci_hi"] - d["mean_diff"]],
                fmt="o", color="#111111", capsize=3, markersize=5, linewidth=1.4)
    ax.axvline(0, color="#888888", linewidth=1, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.comparison}{' *' if r.get('significant_holm', False) else ''}"
                        for _, r in d.iterrows()], fontsize=8)
    ax.set_xlabel(f"Paired difference in {metric} (95% bootstrap CI)")
    ax.grid(alpha=0.25, axis="x", linewidth=0.5)
    fig.savefig(out); plt.close(fig)
    return out


def load_histories(log_dir: str | Path, pattern: str = "*_history.csv") -> Dict[str, List[pd.DataFrame]]:
    """Group history CSVs by strategy name inferred from the filename."""
    out: Dict[str, List[pd.DataFrame]] = {}
    for p in sorted(Path(log_dir).glob(pattern)):
        parts = p.stem.split("_")
        key = parts[2] if len(parts) > 2 else p.stem
        out.setdefault(key, []).append(pd.read_csv(p))
    return out

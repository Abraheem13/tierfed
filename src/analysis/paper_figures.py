"""Manuscript figures (v6): multi-panel, colour, legends below the panels.

Every curve is computed from the per-seed logs in results/; nothing is typed in.
  fig_main.pdf        (a) federated AUROC per round, (b) cumulative upload per
                      site, (c) TierFed intervals and tier dispersion per round
  fig_frontier_dp.pdf (a) accuracy-upload frontier of every run family,
                      (b) matched-privacy AUROC vs epsilon, 20 sites,
                      (c) the same on 100 sites
  fig_emulation.pdf   (a) AUROC against wall-clock time for the wide model on
                      cell-edge NR, (b) emulated vs analytic upload time per run
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .collect import NAMES, load_summaries
from ..network import PROFILES

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,   # embedded TrueType, no Type 3 fonts (IEEE PDF checks)
    "font.size": 7.5, "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "mathtext.fontset": "dejavusans", "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "out", "ytick.direction": "out", "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.linewidth": 0.6, "axes.edgecolor": "#444444", "xtick.color": "#444444", "ytick.color": "#444444",
    "axes.labelcolor": "#111111", "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#e3e3e3",
    "grid.linewidth": 0.5, "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 6.8,
    "savefig.dpi": 300, "savefig.bbox": "tight", "axes.titlesize": 7.5, "axes.labelsize": 7.5,
    "axes.titleweight": "normal", "axes.titlepad": 4,
})

# one colour and line style per method, kept identical across figures
STY = {
    "tierfed":    dict(color="#2b6fd6", ls="-",  lw=2.1, marker="o", label="TierFed (proposed)"),
    "fedper":     dict(color="#f5a623", ls="--", lw=1.2, marker="^", label="FedPer (full rate)"),
    "fedavg":     dict(color="#8c8c8c", ls="-",  lw=1.0, marker="v", label="FedAvg"),
    "adacomm":    dict(color="#2ca02c", ls=":",  lw=1.3, marker="s", label="AdaComm"),
    "fixed":      dict(color="#d9534f", ls="-.", lw=1.2, marker="D", label="Fixed period"),
    "fixed4":     dict(color="#8c564b", ls="--", lw=1.2, marker="D", label="Fixed $K=4$"),
    "fixed8":     dict(color="#9467bd", ls="-.", lw=1.2, marker="P", label="Fixed $K=8$"),
    "tierfed_dp": dict(color="#17becf", ls=":",  lw=1.3, marker="x", label="Planner ($\\lambda=1$)"),
    "lag":        dict(color="#e377c2", ls=":",  lw=1.0, marker="x", label="LAG"),
    "fedlama":    dict(color="#bcbd22", ls=":",  lw=1.0, marker="P", label="FedLAMA-style"),
    "fedprox":    dict(color="#7f7f7f", ls=":",  lw=1.0, marker="h", label="FedProx"),
    "scaffold":   dict(color="#7f7f7f", ls=":",  lw=1.0, marker="*", label="SCAFFOLD"),
}




def _line(ax, x, m, lo, hi, key, **kw):
    st = STY[key]
    ax.plot(x, m, st["ls"], color=st["color"], linewidth=st["lw"], label=kw.get("label", st["label"]),
            solid_capstyle="round", zorder=3 if key == "tierfed" else 2)
    if lo is not None:
        ax.fill_between(x, lo, hi, color=st["color"], alpha=0.13, linewidth=0)


def _align(runs, col, rounds=60):
    """One row per run on a common round grid. History rows are written once
    per round in order (the logged `round` field holds the last synchronisation
    round for multi-rate methods, so position, not that field, is the round);
    early-stopped runs are held at their last value."""
    rows = []
    for r in runs:
        v = pd.Series(r[col].values, index=np.arange(1, len(r) + 1))
        rows.append(v.reindex(range(1, rounds + 1)).ffill().values)
    return np.arange(1, rounds + 1), np.vstack(rows)


def _ci(arr):
    m = np.nanmean(arr, 0)
    se = np.nanstd(arr, 0, ddof=1) / np.sqrt(arr.shape[0])
    return m, m - 1.96 * se, m + 1.96 * se


def _legend_below(fig, axes, ncol, y=0.0):
    h, l = [], []
    for ax in axes:
        hh, ll = ax.get_legend_handles_labels()
        for a_, b_ in zip(hh, ll):
            if b_ not in l:
                h.append(a_); l.append(b_)
    fig.legend(h, l, loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, y), handlelength=2.6,
               handletextpad=0.6, columnspacing=1.6, labelspacing=0.4, borderaxespad=0.0)


# --------------------------------------------------------------------------- #
def fig_main(L: Path, out: Path, key="diabetes_mlp"):
    hist = {}
    for s_ in ["tierfed", "fedper", "fedavg", "adacomm", "fixed"]:
        files = [f for f in sorted(glob.glob(f"{L}/{key}_{s_}_R60_C20_s*_history.csv")) if re.search(r"_s\d+_history", f)]
        if files:
            hist[s_] = [pd.read_csv(f).sort_values("round") for f in files]
    if not hist:
        print("fig_main: no histories"); return
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.15), gridspec_kw={"width_ratios": [1, 1, 1.05]})
    for s_, runs in hist.items():
        x, arr = _align(runs, "fed_auroc"); m, lo, hi = _ci(arr)
        _line(axes[0], x, m, lo, hi, s_)
        x, up = _align(runs, "upload_mb"); m2, _, _ = _ci(up)
        _line(axes[1], x, m2, None, None, s_)
    axes[0].set_title("(a) Federated AUROC (mean, 95% interval)")
    axes[0].set_xlabel("Communication round"); axes[0].set_ylabel("AUROC")
    axes[1].set_title("(b) Cumulative upload per site")
    axes[1].set_xlabel("Communication round"); axes[1].set_ylabel("Upload (MB)")
    tf = hist.get("tierfed")
    if tf is not None and "K_slow" in tf[0]:
        ax = axes[2]
        x, ks = _align(tf, "K_slow"); ks = np.nanmean(ks, 0)
        ax.step(x, ks, where="post", color=STY["tierfed"]["color"], linewidth=2.1, label="Interval $K_g$ (both tiers)", zorder=3)
        ax.set_ylabel("Interval (rounds)"); ax.set_xlabel("Communication round"); ax.set_ylim(0.5, 4.8)
        ax.set_yticks([1, 2, 3, 4])
        ax2 = ax.twinx(); ax2.spines["top"].set_visible(False); ax2.grid(False)
        for col, c, ls, lab in (("divergence_slow", "#8c8c8c", "-", "Dispersion $\\hat\\Gamma_{\\rm s}$ (slow tier)"),
                                ("divergence_medium", "#8c8c8c", "--", "Dispersion $\\hat\\Gamma_{\\rm m}$ (medium tier)")):
            if col in tf[0]:
                _, d = _align(tf, col); d = np.nanmean(d, 0)
                ax2.plot(x, d, ls, color=c, linewidth=1.0, label=lab)
        ax2.set_yscale("log"); ax2.set_ylabel("Dispersion proxy", color="#8c8c8c")
        ax2.tick_params(axis="y", colors="#8c8c8c"); ax2.spines["right"].set_color("#8c8c8c")
        ax.set_title("(c) TierFed intervals and dispersion")
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
        ax._extra_legend = (h1 + h2, l1 + l2)
    # legend: methods first, then the panel-(c) series
    h, l = [], []
    for ax in axes[:2]:
        for a_, b_ in zip(*ax.get_legend_handles_labels()):
            if b_ not in l:
                h.append(a_); l.append(b_)
    if hasattr(axes[2], "_extra_legend"):
        h2, l2 = axes[2]._extra_legend; h += h2; l += l2
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0), handlelength=2.6,
               handletextpad=0.6, columnspacing=1.4, labelspacing=0.4, borderaxespad=0.0)
    fig.tight_layout(rect=(0, 0.17, 1, 1), w_pad=1.6); fig.savefig(out / "fig_main.pdf"); plt.close(fig)


# --------------------------------------------------------------------------- #
def _reached_curve(ax, L: Path, profile="nr_cell_edge"):
    p = PROFILES[profile]
    for s_ in ["fedper", "fedavg", "adacomm", "fixed", "tierfed"]:
        curves = []
        for sj in sorted(L.glob(f"diabetes_mlp_wide_{s_}_R60_C20_s*_wide_summary.json")):
            d = json.loads(sj.read_text())
            hist = pd.read_csv(str(sj).replace("_summary.json", "_history.csv"))
            t = np.cumsum([rb.get("compute_s_max", 0.0) + (rb["max_client_bits"] / (p.uplink_mbps * 1e6) + p.rtt_ms / 1e3 if rb["upload_bits"] > 0 else 0.0)
                           for rb in d["round_bits"]])
            y = pd.Series(hist["fed_auroc"].values, index=np.arange(1, len(hist) + 1)).reindex(range(1, len(t) + 1)).ffill().values
            curves.append((t, y))
        if not curves:
            continue
        grid = np.linspace(0, max(c[0][-1] for c in curves), 300)
        ys = np.vstack([np.interp(grid, c[0], np.maximum.accumulate(c[1]), left=np.nan) for c in curves])
        _line(ax, grid, np.nanmean(ys, 0), None, None, s_)
    ax.set_xlabel("Wall-clock time on cell-edge NR (s)"); ax.set_ylabel("Best AUROC reached"); ax.set_ylim(0.555, 0.65)


def _errline(ax, x, m, e, key, **kw):
    st = STY[key]
    ax.errorbar(x, m, yerr=e, fmt=st["marker"] + st["ls"], color=st["color"], linewidth=st["lw"], elinewidth=0.6,
                capsize=2, capthick=0.6, markersize=3.2, label=kw.get("label", st["label"]),
                zorder=3 if key in ("tierfed", "tierfed_dp") else 2)


def fig_frontier_dp(df: pd.DataFrame, out: Path, dataset="diabetes", model="mlp", L: Path | None = None):
    """One row: (a) AUROC against the quota, (b) matched privacy on 20 sites, (c) on 100 sites, (d) wide model."""
    tag = df["tag"].fillna("")
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.25), gridspec_kw={"width_ratios": [1, 1, 1, 1.05]})
    # (a) quota sweep with 95% intervals; full-rate and fixed-period references
    ax = axes[0]
    main = df[(df.dataset == dataset) & (df.model == model) & (tag == "") & (df["compressor"].fillna("none") == "none")]
    budget = df[tag.str.startswith("_budget_") & (df["solver"].fillna("exact") == "exact")]
    if not budget.empty:
        g = budget.groupby("rho")["fed_auroc_final"].agg(["mean", "std", "count"]).reset_index().sort_values("rho")
        xs = np.arange(len(g))                      # categorical spacing, as for the scenario axis
        _errline(ax, xs, g["mean"], 1.96 * g["std"] / np.sqrt(g["count"]), "tierfed")
        for s_ in ("fedper", "fixed"):
            v = main[main.strategy == s_]["fed_auroc_final"]
            if len(v):
                st = STY[s_]; m, se = v.mean(), v.std(ddof=1) / np.sqrt(len(v))
                ax.axhline(m, color=st["color"], linestyle=st["ls"], linewidth=st["lw"], label=st["label"])
                ax.axhspan(m - 1.96 * se, m + 1.96 * se, color=st["color"], alpha=0.10, linewidth=0)
        ax.set_xticks(xs); ax.set_xticklabels([f"{r:g}" for r in g["rho"]]); ax.set_xlim(-0.4, len(g) - 0.6)
    ax.set_title("(a) Accuracy against the quota"); ax.set_xlabel("Quota $\\rho$"); ax.set_ylabel("Final AUROC")
    # (b), (c) matched privacy
    dpm = df[tag.str.startswith("_dpe") & ~tag.str.contains("_lam")].copy()
    if not dpm.empty:
        dpm["mkey"] = np.where(dpm["strategy"] == "fixed", "fixed" + dpm["k_slow"].fillna(0).astype(float).astype(int).astype(str), dpm["strategy"])
        for ax, N, q, title in ((axes[1], 20, 0.5, "(b) Matched $\\varepsilon$, 20 sites, $q=0.5$"),
                                (axes[2], 100, 0.2, "(c) Matched $\\varepsilon$, 100 sites, $q=0.2$")):
            sub = dpm[(dpm["n_clients"] == N) & (dpm["dp_q"].astype(float) == q)]
            for mk in ["fedavg", "fedper", "fixed4", "fixed8", "tierfed_dp"]:
                g = sub[sub["mkey"] == mk].groupby("dp_target_eps")["fed_auroc_final"].agg(["mean", "std", "count"]).reset_index()
                if g.empty:
                    continue
                _errline(ax, g["dp_target_eps"], g["mean"], 1.96 * g["std"] / np.sqrt(g["count"]), mk)
            ax.axhline(0.5, color="#bbbbbb", linewidth=0.7, linestyle="-", zorder=1)
            ax.set_xscale("log", base=2); ax.set_xticks([1, 2, 4, 8]); ax.set_xticklabels(["1", "2", "4", "8"])
            ax.set_xlabel("Target $\\varepsilon$ ($\\delta=10^{-5}$)"); ax.set_ylabel("Final AUROC"); ax.set_title(title)
    # (d) wide model
    if L is not None:
        _reached_curve(axes[3], L)
        axes[3].set_title("(d) Wide MLP, emulated link")
    _legend_below(fig, axes, ncol=4)
    fig.tight_layout(rect=(0, 0.17, 1, 1), w_pad=1.2); fig.savefig(out / "fig_frontier_dp.pdf"); plt.close(fig)


# --------------------------------------------------------------------------- #
def fig_emulation(R: Path, out: Path, profile="nr_cell_edge"):
    L = R / "logs"; E = R / "emulation"
    emu = [json.loads(p.read_text()) for p in sorted(E.glob("*.json"))] if E.exists() else []
    if not emu:
        print("fig_emulation: no emulation results"); return
    p = PROFILES[profile]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.0), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    for s in ["fedper", "fedavg", "adacomm", "fixed", "tierfed"]:
        curves = []
        for sj in sorted(L.glob(f"diabetes_mlp_wide_{s}_R60_C20_s*_wide_summary.json")):
            d = json.loads(sj.read_text())
            hist = pd.read_csv(str(sj).replace("_summary.json", "_history.csv"))
            t = np.cumsum([rb.get("compute_s_max", 0.0) + (rb["max_client_bits"] / (p.uplink_mbps * 1e6) + p.rtt_ms / 1e3 if rb["upload_bits"] > 0 else 0.0)
                           for rb in d["round_bits"]])
            y = pd.Series(hist["fed_auroc"].values, index=np.arange(1, len(hist) + 1)).reindex(range(1, len(t) + 1)).ffill().values
            curves.append((t, y))
        if not curves:
            continue
        grid = np.linspace(0, max(c[0][-1] for c in curves), 300)
        ys = np.vstack([np.interp(grid, c[0], np.maximum.accumulate(c[1]), left=np.nan) for c in curves])
        _line(ax, grid, np.nanmean(ys, 0), None, None, s, every=40)
    ax.set_xlabel(f"Wall-clock time (s), compute + upload on {profile.replace('_', ' ')}")
    ax.set_ylabel("Federated AUROC"); ax.set_title("(a) Wide MLP (1.74M parameters), best AUROC reached")
    ax.grid(alpha=0.25, linewidth=0.5); ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax = axes[1]
    e = pd.DataFrame(emu)
    for prof, g in e.groupby("profile"):
        ax.scatter(g["analytic_upload_s"], g["emu_upload_s"], s=16, label=prof.replace("_", " "), alpha=0.85)
    lim = [min(e["emu_upload_s"].min(), e["analytic_upload_s"].min()) * 0.8, max(e["emu_upload_s"].max(), e["analytic_upload_s"].max()) * 1.2]
    ax.plot(lim, lim, color="#bbbbbb", linewidth=0.8); ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Analytic upload time (s)"); ax.set_ylabel("Emulated (s), paced TCP")
    ax.set_title("(b) Model against transport, 35 runs"); ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout(); fig.savefig(out / "fig_emulation.pdf"); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results"); ap.add_argument("--out", default="paper/figures")
    a = ap.parse_args(argv)
    R = Path(a.results); O = Path(a.out); O.mkdir(parents=True, exist_ok=True)
    fig_main(R / "logs", O)
    fig_emulation(R, O)
    df = load_summaries(R / "logs")
    if not df.empty:
        fig_frontier_dp(df, O, L=R / "logs")
    print("paper figures written to", O)


if __name__ == "__main__":
    main()

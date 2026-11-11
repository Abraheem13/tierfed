"""Results figures of the v6 runs, drawn from the per-round histories and the
per-round bit logs (nothing is typed in):

  fig_dynamics.pdf  (a) test AUROC per round on the specialty sites, with each
                        method's mean validation-selected round marked;
                    (b) cumulative upload over all sites;
                    (c) one site's upload per round, aligned against staggered
                        periods (seed 42).
  fig_resdp.pdf     (a) test AUROC per round on the ResNet backbone;
                    (b) TierFed's intervals per round on the ResNet;
                    (c) test AUROC per round under client-level DP;
                    (d) the timing observable: total upload of all sites per run,
                        clean against shifted world.

  python -m src.analysis.results_figures --results results --out paper/figures
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .paper_figures import STY, _align, _ci, _legend_below  # noqa: F401  (shared rcParams)

STY["tierfed"] = {**STY["tierfed"], "label": "TierFed", "lw": 1.5}
STY["fedper"] = {**STY["fedper"], "label": "FedPer"}

# styles for series that the shared table does not hold; colours stay distinct from STY
EXTRA = {
    "local":   dict(color="#7f7f7f", ls=":", lw=1.1, label="Local only"),
    "fixed44": dict(color="#8c564b", ls="--", lw=1.2, label="Fixed (4,4)"),
    "fixed88": dict(color="#9467bd", ls="-.", lw=1.2, label="Fixed (8,8)"),
    "fixed15": dict(color="#d9534f", ls="-.", lw=1.2, label="Fixed (1,5)"),
    "oneshot": dict(color="#17becf", ls=(0, (1, 1.2)), lw=1.2, label="One-shot"),
    "fixed22": dict(color="#555555", ls="-.", lw=1.1, label="Fixed (2,2)"),
    "fedavg": dict(color="#555555", ls=(0, (4, 1.5)), lw=1.1, label="FedAvg"),
    "carry": dict(color="#d9534f", ls="-", lw=1.3, label="Fixed (4,4), carry-over"),
    "tierfed_dp": dict(color="#2b6fd6", ls="-.", lw=1.4, label="TierFed-DP planner"),
}


def _sty(key):
    if key in EXTRA:
        return EXTRA[key]
    s = STY[key]
    return dict(color=s["color"], ls=s["ls"], lw=s["lw"], label=s["label"])


def _runs(L: Path, pattern: str):
    files = sorted(glob.glob(str(L / pattern)))
    hist = [pd.read_csv(f) for f in files]
    summ = [json.loads(Path(f.replace("_history.csv", "_summary.json")).read_text()) for f in files]
    return hist, summ


def _curve(ax, hist, col, key, label=None, sel=None, ci=True, lw=None):
    if not hist:
        return None
    x, arr = _align(hist, col)
    m, lo, hi = _ci(arr)
    st = _sty(key)
    ln, = ax.plot(x, m, linestyle=st["ls"], color=st["color"], linewidth=lw or st["lw"], label=label or st["label"],
                  zorder=3 if key in ("tierfed", "tierfed_dp") else 2)
    if ci and arr.shape[0] > 1:
        ax.fill_between(x, lo, hi, color=st["color"], alpha=0.12, linewidth=0)
    if sel is not None and np.isfinite(sel):
        r = int(round(sel))
        ax.plot([r], [m[r - 1]], marker="o", markersize=3.6, color=st["color"], markeredgecolor="white",
                markeredgewidth=0.6, zorder=4)
    return ln


def fig_dynamics(L: Path, out: Path):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 1.9))
    series = [("fedper", "fedper"), ("lag", "lag"), ("adacomm8", "adacomm"), ("tierfed", "tierfed"), ("local", "local")]
    for name, key in series:
        hist, summ = _runs(L, f"diabetes_mlp_*_R60_C20_s*_au_{name}_history.csv")
        sel = np.mean([s["summary"]["valsel_round"] for s in summ]) if summ else np.nan
        _curve(axes[0], hist, "fed_auroc", key, sel=sel)
        if name != "local":
            _curve(axes[1], hist, "upload_mb", key, ci=False)
    axes[0].set_xlabel("Round"); axes[0].set_ylabel("Test AUROC")
    axes[0].set_title("(a) Test AUROC per round")
    axes[1].set_xlabel("Round"); axes[1].set_ylabel("Cumulative upload (MB)")
    axes[1].set_title("(b) Upload of all clients")
    # (c) site 0 of seed 42: upload per round as a share of its full shared payload
    ax = axes[2]
    for name, key, off, lab in (("fixed44", "fixed44", -0.21, "Aligned (4,4)"),
                                ("fixed44_st", "fixed", 0.21, "Staggered (4,4)")):
        f = sorted(glob.glob(str(L / f"diabetes_mlp_fixed_R60_C20_s42_au_{name}_summary.json")))
        if not f:
            continue
        rb = json.loads(Path(f[0]).read_text())["round_bits"]
        full = rb[0]["client_bits"][0]                    # round 1 sends the whole shared payload
        frac = [b["client_bits"][0] / full for b in rb][:21]
        st = _sty(key)
        ax.bar(np.arange(1, len(frac) + 1) + off, frac, width=0.36, color=st["color"], alpha=0.9,
               linewidth=0, label=lab)
    ax.set_xlabel("Round"); ax.set_ylabel("Share of shared payload")
    ax.set_ylim(0, 1.08); ax.set_xlim(0.4, 21.6); ax.set_xticks([1, 5, 9, 13, 17, 21])
    ax.set_title("(c) Upload of one client per round")
    h, l = [], []
    for a in axes:
        for hh, ll in zip(*a.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh); l.append(ll)
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0), handlelength=2.6,
               handletextpad=0.6, columnspacing=1.6, labelspacing=0.4, borderaxespad=0.0)
    fig.tight_layout(rect=(0, 0.19, 1, 1), w_pad=1.0)
    fig.savefig(out / "fig_dynamics.pdf"); plt.close(fig)


def fig_resdp(L: Path, out: Path):
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 1.92))
    # (a) ResNet backbone
    for name, key in (("fedper", "fedper"), ("fixed44", "fixed44"), ("fixed15", "fixed15"), ("tierfed", "tierfed")):
        hist, summ = _runs(L, f"diabetes_resnet_*_R60_C20_s*_au_bb_{name}_history.csv")
        sel = np.mean([s["summary"]["valsel_round"] for s in summ]) if summ else np.nan
        _curve(axes[0], hist, "fed_auroc", key, sel=sel, ci=False, lw=1.1 if key == "tierfed" else None)
    axes[0].set_ylim(0.60, 0.652)
    axes[0].set_xlabel("Round"); axes[0].set_ylabel("Test AUROC"); axes[0].set_title("(a) ResNet, test AUROC")
    # (b) TierFed's intervals on the ResNet
    hist, _ = _runs(L, "diabetes_resnet_tierfed_R60_C20_s*_au_bb_tierfed_history.csv")
    if hist:
        for col, ls, lab in (("K_slow", "-", "$K_{\\mathrm{s}}$ (slow tier)"), ("K_medium", "--", "$K_{\\mathrm{m}}$ (medium tier)")):
            x, arr = _align(hist, col)
            axes[1].plot(x, np.nanmean(arr, 0), linestyle=ls, color=STY["tierfed"]["color"], linewidth=1.4, label=lab)
    axes[1].set_xlabel("Round"); axes[1].set_ylabel("Interval (rounds)")
    axes[1].set_ylim(0, 9.5); axes[1].set_yticks([1, 3, 5]); axes[1].set_title("(b) ResNet, TierFed intervals")
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), fontsize=6.0, handlelength=2.2, labelspacing=0.3)
    # (c) client-level DP at eps* = 4, q = 1
    for name, key in (("fedper", "fedper"), ("fixed44", "fixed44"), ("fixed88", "fixed88"), ("oneshot", "oneshot"),
                      ("tierfed_dp", "tierfed_dp")):
        hist, summ = _runs(L, f"diabetes_mlp_*_R60_C20_s*_au_dpe4_q1_{name}_history.csv")
        _curve(axes[2], hist, "fed_auroc", key)          # same style and label as without DP: the panel says DP
    hist, summ = _runs(L, "diabetes_mlp_local_R60_C20_s*_au_local_history.csv")
    _curve(axes[2], hist, "fed_auroc", "local", ci=False)
    axes[2].set_xlabel("Round"); axes[2].set_ylabel("Test AUROC"); axes[2].set_title("(c) DP, $\\varepsilon^\\star=4$, $q=1$")
    # (d) timing observable: each run's total upload of all clients, relative to the median of the
    # clean-world runs of the same schedule; one row per schedule, symmetric-log axis (linear within +-1 %)
    ax = axes[3]
    order = [("mlp", "fixed44", "Fixed (4,4)"), ("mlp", "tierfed", "TierFed"), ("mlp", "adacomm8", "AdaComm"),
             ("mlp", "lag_xi2000", "LAG, $\\xi$=2000"), ("mlp", "lag_xi5000", "LAG, $\\xi$=5000"),
             ("resnet", "tierfed", "TierFed (ResNet)")]
    order = [o for o in order if glob.glob(str(L / f"diabetes_{o[0]}_*_R60_C20_s*_au_tm_shift_{o[1]}_summary.json"))]
    rng = np.random.default_rng(0)
    ys = np.arange(len(order))[::-1]
    for y, (model, name, lab) in zip(ys, order):
        vals = {}
        for w in ("clean", "shift"):
            fs = sorted(glob.glob(str(L / f"diabetes_{model}_*_R60_C20_s*_au_tm_{w}_{name}_summary.json")))
            vals[w] = np.array([json.loads(Path(f).read_text())["summary"]["total_upload_mb"] for f in fs])
        if not len(vals["clean"]):
            continue
        ref = np.median(vals["clean"])
        for w, off, kw in (("clean", 0.16, dict(marker="o", facecolors="white", edgecolors="#555555")),
                           ("shift", -0.16, dict(marker="o", facecolors="#d9534f", edgecolors="#d9534f"))):
            v = 100 * (vals[w] / ref - 1)
            ax.scatter(v, y + off + rng.uniform(-0.06, 0.06, len(v)), s=7, linewidths=0.5, zorder=3,
                       label=("Clean" if w == "clean" else "Shifted") if y == ys[0] else None, **kw)
    ax.set_xscale("symlog", linthresh=1.0); ax.set_xlim(-60, 3000)
    ax.set_xticks([-10, 0, 10, 1000]); ax.set_xticklabels(["$-$10", "0", "10", "1000"])
    ax.axvline(0, color="#999999", linewidth=0.6, zorder=1)
    ax.set_yticks(ys); ax.set_yticklabels([o[2] for o in order], fontsize=6.2)
    ax.tick_params(axis="y", length=0); ax.set_ylim(-0.6, len(order) - 0.4)
    ax.grid(False); ax.grid(axis="x", color="#e3e3e3", linewidth=0.5)
    k = [i for i, o in enumerate(order) if o[0] == "resnet"]
    if k:
        ax.axhline(ys[k[0]] + 0.5, color="#bbbbbb", linewidth=0.6, linestyle=(0, (2, 2)), zorder=1)
    ax.set_xlabel("Upload vs. clean median (%)")
    ax.set_title("(d) Traffic seen by an observer")
    ax.legend(loc="upper right", fontsize=6.0, handletextpad=0.1, borderaxespad=0.1, labelspacing=0.25)
    h, l = [], []
    for a in axes[[0, 2]]:
        for hh, ll in zip(*a.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh); l.append(ll)
    fig.legend(h, l, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.0), handlelength=2.6,
               handletextpad=0.6, columnspacing=1.4, labelspacing=0.4, borderaxespad=0.0)
    fig.tight_layout(rect=(0, 0.2, 1, 1), w_pad=0.8)
    fig.savefig(out / "fig_resdp.pdf"); plt.close(fig)


def fig_sync(L: Path, out: Path):
    """(a, b) the four clinical centres: test AUROC per round and cumulative upload;
    (c) the fixed period (4,4) on the specialty groups under both synchronisation steps;
    (d) dispersion of the uploaded slices against the uniform period, both steps."""
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 1.92))
    for name, key in (("fedper", "fedper"), ("fixed22", "fixed22"), ("tierfed", "tierfed"), ("adacomm8", "adacomm"),
                      ("local", "local")):
        hist, summ = _runs(L, f"heart_mlp_*_R60_C4_s*_au_hosp_{name}_history.csv")
        sel = np.mean([s["summary"]["valsel_round"] for s in summ]) if summ else np.nan
        _curve(axes[0], hist, "fed_auroc", key, sel=sel, ci=False)
        if name != "local":
            _curve(axes[1], hist, "upload_mb", key, ci=False)
    axes[0].set_xlabel("Round"); axes[0].set_ylabel("Test AUROC"); axes[0].set_title("(a) Clinical centres, AUROC")
    axes[1].set_xlabel("Round"); axes[1].set_ylabel("Cumulative upload (MB)"); axes[1].set_title("(b) Clinical centres, upload")
    # (c) restart against carry-over
    for pat, key, lab in (("diabetes_mlp_fedper_R60_C20_s*_au_fedper_history.csv", "fedper", None),
                          ("diabetes_mlp_fixed_R60_C20_s*_au_fixed44_history.csv", "fixed44", "Fixed (4,4), restart"),
                          ("diabetes_mlp_fixed_R60_C20_s*_au_co_fixed44_history.csv", "carry", None)):
        hist, summ = _runs(L, pat)
        sel = np.mean([s["summary"]["valsel_round"] for s in summ]) if summ else np.nan
        _curve(axes[2], hist, "fed_auroc", key, label=lab, sel=sel)
    axes[2].set_xlabel("Round"); axes[2].set_ylabel("Test AUROC"); axes[2].set_title("(c) Synchronisation step")
    # (d) dispersion against the period
    ax = axes[3]
    for step, pre, ls, mk in (("restart", "", "-", "o"), ("carry-over", "co_", "--", "s")):
        for tier, col in (("slow", "#222222"), ("medium", "#999999")):
            xs, ys = [], []
            for k, strat, name in ((1, "fedper", "fedper"), (2, "fixed", "fixed22"), (4, "fixed", "fixed44"),
                                   (6, "fixed", "fixed66"), (8, "fixed", "fixed88")):
                if pre and k == 1:
                    continue
                v = []
                for f in sorted(glob.glob(str(L / f"diabetes_mlp_{strat}_R60_C20_s*_au_{pre}{name}_history.csv"))):
                    h = pd.read_csv(f); tx = h[(h["round"] > 5) & (h["payload_fraction"] > 0)]
                    if len(tx):
                        v.append(tx[f"divergence_{tier}"].mean())
                if v:
                    xs.append(k); ys.append(np.mean(v))
            if xs:
                ax.plot(xs, ys, linestyle=ls, marker="o" if tier == "slow" else "s", color=col, markersize=3.0,
                        linewidth=1.1, label=f"{tier.capitalize()} tier, {step}")
    ax.set_yscale("log"); ax.set_xticks([1, 2, 4, 6, 8]); ax.set_xlabel("Uniform period $K$")
    ax.set_ylabel("Dispersion $\\hat\\Gamma_g$"); ax.set_title("(d) Dispersion of uploads")
    ax.set_ylim(2e-3, 0.4)
    h, l = [], []
    for a in axes:
        for hh, ll in zip(*a.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh); l.append(ll)
    fig.legend(h, l, loc="lower center", ncol=6, bbox_to_anchor=(0.5, 0.0), handlelength=2.6,
               handletextpad=0.5, columnspacing=1.2, labelspacing=0.4, borderaxespad=0.0)
    fig.tight_layout(rect=(0, 0.2, 1, 1), w_pad=0.8)
    fig.savefig(out / "fig_sync.pdf"); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results"); ap.add_argument("--out", default="paper/figures")
    a = ap.parse_args(argv)
    L = Path(a.results) / "logs"; out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fig_dynamics(L, out); fig_resdp(L, out); fig_sync(L, out)
    print("written", out / "fig_dynamics.pdf", out / "fig_resdp.pdf", out / "fig_sync.pdf")


if __name__ == "__main__":
    main()

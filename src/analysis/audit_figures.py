"""v6 result figures in the manuscript's house style (see paper_figures.py).

  python -m src.analysis.audit_figures --results results --out paper/figures

fig_au.pdf, one row of four panels:
  (a) accuracy against upload for every method with its own knob swept
  (b) accuracy against label skew (Dirichlet alpha)
  (c) gain over the best uniform period at matched bytes against Psi (Prop. 2)
  (d) window violations against the window, aligned against staggered tiers
All accuracies are test AUROC at the validation-selected round, mean and 95% CI over seeds.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .audit_report import MATCH_TOL, PRIMARY, _uniform_k, best_uniform, family, gate_table, load, paired
from .paper_figures import STY, _legend_below  # noqa: F401  (imports the shared rcParams)

FAM_STY = {"TierFed": "tierfed", "Uniform period": "fixed", "AdaComm": "adacomm", "LAG": "lag",
           "FedLAMA": "fedlama"}
SET_LAB = {"specialty": "spec.", "a0.1": "$\\alpha$=0.1", "a0.3": "$\\alpha$=0.3", "a1": "$\\alpha$=1",
           "a10": "$\\alpha$=10", "resnet": "ResNet", "transformer": "Transformer"}


def _mci(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan, np.nan
    return v.mean(), (1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0)


def _key_of(name):
    for k in ("knob",):
        pass
    return name


def panel_frontier(ax, df):
    sp = df[(df.setting == "specialty") & df.eps.isna()].copy()
    sp["family"] = sp.name.map(family)
    for fam, key in FAM_STY.items():
        f = sp[sp.family == fam]
        if f.empty:
            continue
        g = f.groupby("name").agg(mb=("total_upload_mb", "mean")).join(
            f.groupby("name")[PRIMARY].agg(lambda v: _mci(v)[0]).rename("m")).join(
            f.groupby("name")[PRIMARY].agg(lambda v: _mci(v)[1]).rename("e")).sort_values("mb")
        g = g[g.mb > 0]
        st = STY[key]
        ax.errorbar(g.mb, g.m, yerr=g.e, fmt=st["marker"] + st["ls"], color=st["color"], linewidth=st["lw"],
                    elinewidth=0.6, capsize=2, capthick=0.6, markersize=3.0,
                    label="Uniform period" if fam == "Uniform period" else st["label"],
                    zorder=3 if key == "tierfed" else 2)
    for n, lab, ls in (("local", "Local only", ":"), ("central", "Centralised", "--"), ("hgb", "Gradient boosting", "-.")):
        v = sp[sp.name == n][PRIMARY]
        if len(v):
            ax.axhline(v.mean(), color="#999999", linestyle=ls, linewidth=0.8, label=lab, zorder=1)
    ax.set_xscale("log"); ax.set_xlabel("Upload, total over sites (MB)"); ax.set_ylabel("AUROC")
    ax.set_title("(a) Accuracy against upload")


def panel_hetero(ax, df):
    alphas = sorted([s for s in df.setting.unique() if s.startswith("a")], key=lambda s: float(s[1:]))
    if not alphas:
        ax.set_title("(b) Accuracy against label skew"); return
    xs = np.arange(len(alphas))
    series = {"tierfed": [], "fixed": [], "fedper": [], "fedavg": []}
    for st_ in alphas:
        d = df[(df.setting == st_) & df.eps.isna()]
        tf_mb = d[d.name == "tierfed"].total_upload_mb.mean()
        bu = best_uniform(d, MATCH_TOL * tf_mb) if np.isfinite(tf_mb) else None
        for key, name in (("tierfed", "tierfed"), ("fixed", bu), ("fedper", "fedper"), ("fedavg", "fedavg")):
            series[key].append(_mci(d[d.name == name][PRIMARY]) if name else (np.nan, np.nan))
    for key, vals in series.items():
        m = np.array([v[0] for v in vals]); e = np.array([v[1] for v in vals])
        st = STY[key]
        ax.errorbar(xs, m, yerr=e, fmt=st["marker"] + st["ls"], color=st["color"], linewidth=st["lw"],
                    elinewidth=0.6, capsize=2, capthick=0.6, markersize=3.0,
                    label="Best uniform, matched bytes" if key == "fixed" else st["label"],
                    zorder=3 if key == "tierfed" else 2)
    loc = [_mci(df[(df.setting == s) & (df.name == "local")][PRIMARY])[0] for s in alphas]
    ax.plot(xs, loc, ":", color="#999999", linewidth=0.8)
    ax.set_xticks(xs); ax.set_xticklabels([s[1:] for s in alphas]); ax.set_xlim(-0.4, len(xs) - 0.6)
    ax.set_xlabel("Dirichlet $\\alpha$ (smaller: more skew)"); ax.set_ylabel("AUROC")
    ax.set_title("(b) Accuracy against label skew")


def panel_psi(ax, df, gate):
    st = STY["tierfed"]
    for _, r in gate.iterrows():
        d = df[(df.setting == r.setting) & (df.name == "tierfed")]
        psi = d["psi"].mean() if "psi" in d else np.nan
        if not np.isfinite(psi) or not np.isfinite(r.gain_vs_uniform):
            continue
        x = max(100 * psi, 0.01)
        ax.errorbar([x], [r.gain_vs_uniform], yerr=[[r.gain_vs_uniform - r.gain_lo], [r.gain_hi - r.gain_vs_uniform]],
                    fmt="o", color=st["color"], markersize=3.4, elinewidth=0.7, capsize=2, capthick=0.6, zorder=3)
        ax.annotate(SET_LAB.get(r.setting, r.setting), (x, r.gain_vs_uniform), textcoords="offset points",
                    xytext=(4, 3), fontsize=6.2, color="#333333")
    ax.axhline(0, color="#999999", linewidth=0.7)
    ax.set_xscale("log"); ax.set_xlabel("$\\Psi$ at the end of warm-up (%)")
    ax.set_ylabel("Gain over best uniform (AUROC)")
    ax.set_title("(c) Gain against $\\Psi$")


def panel_peak(ax, df, uplink_mbps=5.0, rtt_s=0.03):
    sp = df[(df.setting == "specialty") & df.eps.isna()]
    fr = np.linspace(0.3, 1.1, 33)
    for name, key, lab in (("fixed44", "fixed4", "Fixed (4,4), aligned"),
                           ("fixed44_st_w8", "fixed8", "Fixed (4,4), staggered, 8-bit warm-up"),
                           ("tierfed", "tierfed", "TierFed"), ("tierfed_st_w8", "tierfed_dp", "TierFed, staggered, 8-bit warm-up")):
        runs = sp[sp.name == name]
        if runs.empty:
            continue
        curves = []
        for _, r in runs.iterrows():
            rb = r["_round_bits"]; sb = r.get("shared_bits")
            if not rb or not sb:
                continue
            t1 = sb / (uplink_mbps * 1e6) + rtt_s
            t = np.array([b["max_client_bits"] / (uplink_mbps * 1e6) + rtt_s for b in rb if b["upload_bits"] > 0])
            curves.append([(t > f * t1 + 1e-12).sum() / len(rb) for f in fr])
        if curves:
            st = STY[key]
            ax.plot(fr, np.mean(curves, 0), st["ls"], color=st["color"], linewidth=st["lw"], label=lab)
    ax.set_xlabel("Window $W$ / full-payload upload time $t_1$"); ax.set_ylabel("Rounds over the window")
    ax.set_title("(d) Window violations, 5 Mbit/s")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results"); ap.add_argument("--out", default="paper/figures")
    ap.add_argument("--dataset", default="diabetes")
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    df = load(a.results, a.dataset)
    df = df[(df.setting != "derma") & ~df.setting.str.startswith("tm_")]
    gate = gate_table(df)
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.35))
    panel_frontier(axes[0], df); panel_hetero(axes[1], df); panel_psi(axes[2], df, gate); panel_peak(axes[3], df)
    _legend_below(fig, axes, ncol=4)
    fig.tight_layout(rect=(0, 0.2, 1, 1), w_pad=1.0)
    fig.savefig(out / "fig_au.pdf"); plt.close(fig)
    print("written", out / "fig_au.pdf")


if __name__ == "__main__":
    main()

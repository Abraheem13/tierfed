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

STY["tierfed"] = {**STY["tierfed"], "label": "TierFed", "lw": 1.6}
STY["fedper"] = {**STY["fedper"], "label": "FedPer"}

FAM_STY = {"TierFed": "tierfed", "Uniform period": "fixed", "AdaComm": "adacomm", "LAG": "lag",
           "FedLAMA": "fedlama"}
SET_LAB = {"specialty": "spec.", "a0.1": "$\\alpha$=0.1", "a0.3": "$\\alpha$=0.3", "a1": "$\\alpha$=1",
           "a10": "$\\alpha$=10", "resnet": "ResNet", "transformer": "Transformer", "hospitals": "centres"}


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
    from .audit_report import common_seeds
    sp = df[(df.setting == "specialty") & df.eps.isna()].copy()
    sp = sp[sp.seed.isin(common_seeds(sp[~sp.name.isin(["central", "hgb"])]))]
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
    ax.set_xscale("log")
    lo = sp.loc[sp.total_upload_mb > 0, "total_upload_mb"].min()
    ticks = [t for t in (3, 10, 30, 100) if t >= 0.7 * lo]
    ax.set_xticks(ticks); ax.set_xticklabels([str(t) for t in ticks]); ax.minorticks_off()
    ax.set_xlim(0.7 * min(ticks[0], lo), 300)
    ax.set_xlabel("Upload over 60 rounds (MB)"); ax.set_ylabel("AUROC")
    ax.set_title("(a) Accuracy against upload")


def panel_hetero(ax, df):
    alphas = sorted([s for s in df.setting.unique() if s.startswith("a")], key=lambda s: float(s[1:]))
    if not alphas:
        ax.set_title("(b) Label skew"); return
    xs = np.arange(len(alphas))
    series = {"tierfed": [], "fixed": [], "fedper": [], "fedavg": []}
    for st_ in alphas:
        d = df[(df.setting == st_) & df.eps.isna()]
        tf_mb = d[d.name == "tierfed"].total_upload_mb.mean()
        bu = best_uniform(d, MATCH_TOL * tf_mb) if np.isfinite(tf_mb) else None
        for key, name in (("tierfed", "tierfed"), ("fixed", bu), ("fedper", "fedper"), ("fedavg", "fedavg")):
            series[key].append(_mci(d[d.name == name][PRIMARY]) if name else (np.nan, np.nan))
    dodge = {"tierfed": -0.15, "fixed": -0.05, "fedper": 0.05, "fedavg": 0.15}   # identical series stay visible
    for key, vals in series.items():
        m = np.array([v[0] for v in vals]); e = np.array([v[1] for v in vals])
        if not np.isfinite(m).any():
            continue                  # a method that was not run in this sweep gets no legend entry
        st = STY[key]
        ax.errorbar(xs + dodge[key], m, yerr=e, fmt=st["marker"] + st["ls"], color=st["color"], linewidth=st["lw"],
                    elinewidth=0.6, capsize=2, capthick=0.6, markersize=3.0,
                    label="Uniform period" if key == "fixed" else st["label"],
                    zorder=3 if key == "tierfed" else 2)
    loc = [_mci(df[(df.setting == s) & (df.name == "local")][PRIMARY])[0] for s in alphas]
    ax.plot(xs, loc, ":", color="#999999", linewidth=0.8)
    ax.set_xticks(xs); ax.set_xticklabels([s[1:] for s in alphas]); ax.set_xlim(-0.4, len(xs) - 0.6)
    ax.set_xlabel("Dirichlet $\\alpha$"); ax.set_ylabel("AUROC")
    ax.set_title("(b) Label skew")


SET_NAME = {"specialty": "Specialty", "a0.1": "$\\alpha=0.1$", "a0.3": "$\\alpha=0.3$", "a1": "$\\alpha=1$",
            "a10": "$\\alpha=10$", "resnet": "ResNet", "hospitals": "Centres"}


def panel_psi(ax, df, gate):
    """Gain over the best uniform period at matched bytes in every setting, with its 95%
    interval; settings are ordered by Psi, which the right-hand axis lists. Open markers:
    TierFed chose the best uniform period and the runs are identical (gain exactly 0)."""
    from .audit_report import PRACTICAL
    st = STY["tierfed"]
    rows = []
    for _, r in gate.iterrows():
        d = df[(df.setting == r.setting) & (df.name == "tierfed")]
        psi = d["psi"].mean() if "psi" in d else np.nan
        if np.isfinite(psi) and np.isfinite(r.gain_vs_uniform):
            same = bool(abs(r.gain_vs_uniform) < 1e-12 and abs(r.gain_hi - r.gain_lo) < 1e-12)
            rows.append((100 * psi, r.setting, r.gain_vs_uniform, r.gain_lo, r.gain_hi, same))
    rows.sort(key=lambda t: t[0])
    ys = np.arange(len(rows))[::-1]
    ax.axvspan(-PRACTICAL, PRACTICAL, color="#eeeeee", zorder=0, linewidth=0)
    ax.axvline(0, color="#888888", linewidth=0.7, zorder=1)
    for y, (psi, name, g, lo, hi, same) in zip(ys, rows):
        if same:
            ax.plot([g], [y], "o", markersize=3.6, markerfacecolor="white", markeredgecolor=st["color"],
                    markeredgewidth=0.9, zorder=3)
        else:
            ax.errorbar([g], [y], xerr=[[g - lo], [hi - g]], fmt="o", color=st["color"], markersize=3.6,
                        elinewidth=0.9, capsize=1.8, capthick=0.7, zorder=3)
    ax.set_yticks(ys); ax.set_yticklabels([SET_NAME.get(t[1], t[1]) for t in rows])
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.grid(False); ax.grid(axis="x", color="#e3e3e3", linewidth=0.5)
    ax.set_xlim(-0.03, 0.012); ax.set_xticks([-0.02, 0.0, 0.01])
    ax.set_xticklabels(["$-$0.02", "0", "0.01"])
    ax.set_xlabel("$\\Delta_{\\mathrm{u}}$ (AUROC)")
    right = ax.twinx()
    right.set_ylim(ax.get_ylim()); right.set_yticks(ys)
    right.set_yticklabels([f"{t[0]:.1f}" for t in rows]); right.grid(False)
    right.spines["right"].set_visible(True); right.spines["top"].set_visible(False)
    right.tick_params(axis="y", length=0, pad=2, labelsize=6.4)
    right.set_ylabel("$\\Psi$ (%)", labelpad=2)
    ax.tick_params(axis="y", length=0, labelsize=6.4)
    ax.set_title("(c) Gain over best uniform period")


def panel_peak(ax, df, uplink_mbps=5.0, rtt_s=0.03):
    """Share of rounds whose upload time exceeds a window W, against W relative to the upload
    time of the full shared payload. TierFed produces the same curves as the fixed period with
    the same phases (identical runs), so each schedule is drawn once."""
    sp = df[(df.setting == "specialty") & df.eps.isna()]
    fr = np.linspace(0.25, 1.15, 901)
    series = (("fixed44", "#8c564b", "-", 1.4, "Aligned (4,4)"),
              ("fixed44_st", "#d9534f", "--", 1.2, "Staggered (4,4)"),
              ("fixed44_st_w8", "#555555", ":", 1.3, "Staggered, 8-bit warm-up"))
    for name, col, ls, lw, lab in series:
        runs = sp[sp.name == name]
        curves = []
        for _, r in runs.iterrows():
            rb = r["_round_bits"]; sb = r.get("shared_bits")
            if not rb or not sb:
                continue
            t1 = sb / (uplink_mbps * 1e6) + rtt_s
            t = np.array([b["max_client_bits"] / (uplink_mbps * 1e6) + rtt_s for b in rb if b["upload_bits"] > 0])
            curves.append([(t > f * t1 + 1e-12).sum() / len(rb) for f in fr])
        if curves:
            ax.plot(fr, np.mean(curves, 0), linestyle=ls, color=col, linewidth=lw, label=lab)
    ax.set_xlim(0.25, 1.15); ax.set_ylim(-0.02, 0.62)
    ax.set_xlabel("Window / full-payload upload time"); ax.set_ylabel("Share of rounds")
    ax.set_title("(d) Missed transfer windows")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results"); ap.add_argument("--out", default="paper/figures")
    ap.add_argument("--dataset", default="diabetes")
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    df = load(a.results, a.dataset)
    df = df[(df.setting != "derma") & ~df.setting.str.startswith("tm_") & ~df.setting.str.endswith("_co")]
    gate = gate_table(df)
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.0))
    panel_frontier(axes[0], df); panel_hetero(axes[1], df); panel_psi(axes[2], df, gate); panel_peak(axes[3], df)
    _legend_below(fig, [axes[0], axes[1], axes[3]], ncol=6)   # (c) needs none
    fig.tight_layout(rect=(0, 0.17, 1, 1), w_pad=1.6)
    fig.savefig(out / "fig_au.pdf"); plt.close(fig)
    print("written", out / "fig_au.pdf")


if __name__ == "__main__":
    main()

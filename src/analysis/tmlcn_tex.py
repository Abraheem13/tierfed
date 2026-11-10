"""Tables, numbers and checked statements of the TMLCN manuscript, from the run logs.

  python -m src.analysis.tmlcn_tex --results results --out paper/tmlcn/generated

Writes into --out
  numbers_au.tex, au_*.tex   the output of src.analysis.audit_tex (all families)
  numbers_tmlcn.tex          numbers and true/false flags used only by this manuscript
  tm_*.tex                   the tables in the manuscript's layout
Nothing is typed by hand: the manuscript source (paper/tmlcn/src) refers to these
files, and scripts/tmlcn/build.py refuses to build when a statement the prose makes
is not supported by its flag (paper/tmlcn/claims.json).
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import audit_tex
from .audit_report import (MAIN_ORDER, MATCH_TOL, PRACTICAL, PRIMARY, _uniform_k, best_uniform, common_seeds,
                           dp_table, gate_table, holm, load, main_table, paired, peak_table, timing_table)
from .audit_tex import NAMES, SETTING, early_k, f4, fp, fs, k_label
from .collect import TexWriter, _safe

SETTING = {**SETTING, "specialty": "Specialty groups", "hospitals": "Clinical centres"}
NAMES = {**NAMES, "fixed44_st": "Staggered (4,4)", "fixed44_st_w8": "Staggered (4,4), 8-bit warm-up",
         "fixed15": "Fixed (1,5)", "fixed22": "Fixed (2,2)", "adacomm8": "AdaComm"}
HOSP_ORDER = ["local", "fedavg", "fedprox", "fedper", "fedlama", "adacomm8", "lag", "fixed22", "fixed88",
              "fixed44", "fixed44_st", "tierfed", "tierfed_st"]
RULES = (("argmax", "First maximum of $A^{\\mathcal V}$ (main text)"),
         ("smooth3", "Maximum of the three-round mean of $A^{\\mathcal V}$"),
         ("last10", "Mean of the last ten rounds"))


# --------------------------------------------------------------------------- #
def ci(v):
    return audit_tex.ci_mean(v)


WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve")


def word(n: int) -> str:
    """Small counts are written as words in running text."""
    return WORDS[n] if 0 <= int(n) < len(WORDS) else str(int(n))


def identical(d: pd.DataFrame, a: str, b: str) -> bool:
    """Two configurations gave the same run on every seed they share (same selected
    round, same test AUROC there and at the end, same bytes)."""
    x = d[d.name == a].set_index("seed"); y = d[d.name == b].set_index("seed")
    common = x.index.intersection(y.index)
    if not len(common) or len(common) != len(x) or len(common) != len(y):
        return False
    cols = [PRIMARY, "valsel_round", "fed_auroc_final", "total_upload_mb"]
    return bool((x.loc[common, cols].to_numpy() == y.loc[common, cols].to_numpy()).all())


def merged_rows(d, order):
    """[(label, representative name)]: TierFed is listed together with a fixed period
    when the two gave identical runs."""
    names = [n for n in order if n in set(d.name)]
    out, skip = [], set()
    for tf, fx in (("tierfed", "fixed44"), ("tierfed_st", "fixed44_st"), ("tierfed_st_w8", "fixed44_st_w8")):
        if tf in names and fx in names and identical(d, tf, fx):
            skip.add(fx)
    for n in names:
        if n in skip:
            continue
        lab = NAMES.get(n, n)
        for tf, fx in (("tierfed", "fixed44"), ("tierfed_st", "fixed44_st"), ("tierfed_st_w8", "fixed44_st_w8")):
            if n == tf and fx in skip:
                lab = {"tierfed": "\\tierfed{} $\\equiv$ Fixed (4,4)",
                       "tierfed_st": "\\tierfed{}, staggered $\\equiv$ Staggered (4,4)",
                       "tierfed_st_w8": "\\tierfed{} $\\equiv$ Staggered (4,4), 8-bit warm-up"}[tf]
        out.append((lab, n))
    return out


def tabular(spec, head, rows):
    return (f"\\begin{{tabular}}{{{spec}}}\n\\toprule\n{head}\\\\\n\\midrule\n" + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabular}\n")


# --------------------------------------------------------------------------- #
def tab_main(d, mt, order, with_ft=True, worst="Worst site"):
    rows = []
    for lab, n in merged_rows(d, order):
        if n not in mt.index:
            continue
        r = mt.loc[n]; lo, hi = ci(d[d.name == n][PRIMARY])
        bound = n in ("central", "hgb")
        if bound and "\\midrule" not in rows:
            rows.append("\\midrule")
        cells = [lab, f"{f4(r.get(PRIMARY + '_mean'))} [{f4(lo)}, {f4(hi)}]",
                 "--" if bound else f4(r.get("upload_mb_valsel_mean"), 1),
                 "--" if bound else f4(r.get("total_upload_mb_mean"), 1),
                 f4(r.get("fed_auroc_last10_mean_mean"))]
        if with_ft:
            cells += [f4(r.get("fed_auroc_phase_mean")), f4(r.get("fed_ft3_auroc_final_mean")),
                      f"{f4(r.get('fed_ece_valsel_mean'), 3)}/{f4(r.get('fed_cal_ece_valsel_mean'), 3)}"]
        cells += [f4(r.get("fed_auroc_site_min_valsel_mean"), 3),
                  "--" if n == "fedper" else fs(r.get("d_fedper_valsel", np.nan)),
                  "--" if n == "fedper" else fp(r.get("p_fedper_valsel"))]
        rows.append(" & ".join(cells) + "\\\\")
    if with_ft:
        return tabular("@{}lcccccccccc@{}",
                       "Method & AUROC at $\\hat r$ [95\\% CI] & MB to $\\hat r$ & MB, 60 rounds & Last 10 & Phase "
                       "& +FT 3 & ECE raw/Platt & Worst site & $\\Delta$ FedPer & $p$", rows)
    return tabular("@{}lccccccc@{}",
                   "Method & AUROC at $\\hat r$ [95\\% CI] & MB to $\\hat r$ & MB, 60 rounds & Last 10 "
                   f"& {worst} & $\\Delta$ FedPer & $p$", rows)


def paired_rows(d, order):
    """TierFed against every other configuration of the comparison, paired over seeds."""
    tf = d[d.name == "tierfed"].set_index("seed")
    rows = []
    for lab, n in merged_rows(d, order):
        if n == "tierfed" or n in ("central", "hgb"):
            continue
        r = paired(d, "tierfed", n)
        b = d[d.name == n].set_index("seed"); common = tf.index.intersection(b.index)
        rows.append(dict(name=n, label=NAMES.get(n, n) if n != "tierfed_st" else "Staggered (4,4)", **r,
                         higher=int((tf.loc[common, PRIMARY] > b.loc[common, PRIMARY]).sum()), k=len(common),
                         dmb=float(tf.loc[common, "total_upload_mb"].mean() - b.loc[common, "total_upload_mb"].mean()),
                         dmb_sel=float(tf.loc[common, "upload_mb_valsel"].mean() - b.loc[common, "upload_mb_valsel"].mean())))
    adj = holm([r["p"] for r in rows]) if rows else []
    for r, ph in zip(rows, adj):
        r["p_holm"] = float(ph)
    return rows


def tab_paired(rows):
    body = [" & ".join([r["label"], f"{fs(r['diff'])} [{fs(r['lo'])}, {fs(r['hi'])}]", fp(r["p_holm"]),
                        f"{r['higher']}/{r['k']}", fs(r["dmb_sel"], 1), fs(r["dmb"], 1)]) + "\\\\" for r in rows]
    return tabular("@{}lccccc@{}", "\\tierfed{} against & $\\Delta$AUROC [95\\% CI] & $p_{\\mathrm{Holm}}$ & Higher "
                   "& $\\Delta$MB to $\\hat r$ & $\\Delta$MB, 60 rounds", body)


def tab_gate(base, gate):
    rows = []
    for _, r in gate.iterrows():
        d = base[(base.setting == r.setting) & base.eps.isna()]
        t = d[d.name == "tierfed"]
        psi = t["psi"].mean() if "psi" in t else np.nan
        same = bool(r.best_uniform) and identical(d, "tierfed", r.best_uniform)
        gain = ("identical runs" if same else
                f"{fs(r.gain_vs_uniform)} [{fs(r.gain_lo)}, {fs(r.gain_hi)}]" if np.isfinite(r.gain_vs_uniform) else "--")
        rows.append(" & ".join([
            SETTING.get(r.setting, r.setting), f4(100 * psi, 1) if np.isfinite(psi) else "--", early_k(t),
            f4(r.tierfed_auroc), f4(r.tierfed_mb, 0), k_label(r.best_uniform) if r.best_uniform else "--",
            f4(r.best_uniform_auroc), f4(r.best_uniform_mb, 0), gain,
            "--" if same else fp(r.get("p_vs_uniform", np.nan)), str(int(r.get("n_pairs", 0)))]) + "\\\\")
    return tabular("@{}lcccccccccc@{}",
                   "Setting & $\\Psi$ (\\%) & $\\mathbf K$ & \\tierfed{} & MB & Best uniform & AUROC & MB "
                   "& Gain $\\Delta_{\\mathrm u}$ [95\\% CI] & $p$ & Seeds", rows)


def tab_network(sp, pk):
    order = ["fedper", "adacomm8", "lag", "fixed88", "fixed44", "tierfed", "fixed44_st", "tierfed_st",
             "fixed44_st_w8", "tierfed_st_w8"]
    rows = []
    for lab, n in merged_rows(sp, order):
        if n in ("tierfed_st", "tierfed_st_w8"):
            lab = NAMES["fixed44_st"] if n == "tierfed_st" else NAMES["fixed44_st_w8"]
        d = sp[sp.name == n]
        g = lambda c, nd=0: f4(d[c].mean(), nd) if c in d and d[c].notna().any() else "--"   # noqa: E731
        post = pk.loc[n, "peak_post_frac"] if n in pk.index else np.nan
        viol = pk.loc[n, "violation_rate_W0.6"] if n in pk.index else np.nan
        rows.append(" & ".join([lab, g("total_upload_mb", 1), f4(post, 2), f4(viol, 2), g("wakeups"),
                                g("net_nr_cell_edge_comm_time_s", 1), g("net_nr_cell_edge_shared_comm_time_s"),
                                g("net_nr_cell_edge_energy_j", 1), g("net_leo_sat_shared_comm_time_s"),
                                g("net_leo_sat_energy_j", 0), g("net_hospital_wan_shared_comm_time_s", 1)]) + "\\\\")
    head = (" & & & & & \\multicolumn{3}{c}{5G NR, cell edge} & \\multicolumn{2}{c}{LEO satellite} & WAN\\\\\n"
            "\\cmidrule(lr){6-8}\\cmidrule(lr){9-10}\\cmidrule(lr){11-11}\n"
            "Method & MB & Peak & Missed & Wake-ups & $t$ (s) & $t_{\\mathrm{sh}}$ (s) & $E$ (J) "
            "& $t_{\\mathrm{sh}}$ (s) & $E$ (J) & $t_{\\mathrm{sh}}$ (s)")
    return tabular("@{}lcccccccccc@{}", head, rows)


DP_ARMS = ["fedper", "fixed44", "fixed88", "oneshot", "tierfed_dp"]
DP_NAMES = {"fedper": "FedPer", "fixed44": "Fixed (4,4)", "fixed88": "Fixed (8,8)", "oneshot": "One-shot",
            "tierfed_dp": "\\tierfed{}-DP"}


def dp_frames(df):
    d = df[(df.eps == 4.0) & (df.q == 1.0) & df.name.isin(DP_ARMS)]
    seeds = sorted(set.intersection(*[set(d[d.name == n].seed) for n in DP_ARMS if n in set(d.name)])) if len(d) else []
    d = d[d.seed.isin(seeds)]
    loc = df[(df.setting == "specialty") & (df.name == "local") & df.eps.isna() & df.seed.isin(seeds)]
    return d, loc, seeds


def tab_dp(df):
    d, loc, seeds = dp_frames(df)
    rows = []
    for n in [a for a in DP_ARMS if a in set(d.name)]:
        x = d[d.name == n]
        rows.append(" & ".join([DP_NAMES[n], f4(x["dp_rounds_charged"].mean(), 0), f4(x["dp_sigma"].mean(), 2),
                                f4(x["clip_fraction"].mean(), 2), f4(x["fed_auroc_final"].mean()),
                                f4(x[PRIMARY].mean()), f4(x["valsel_round"].mean(), 0)]) + "\\\\")
    rows.append("\\midrule")
    rows.append(" & ".join(["Local training", "0", "--", "--", f4(loc["fed_auroc_final"].mean()),
                            f4(loc[PRIMARY].mean()), f4(loc["valsel_round"].mean(), 0)]) + "\\\\")
    return tabular("@{}lcccccc@{}", "Method & Releases & $z$ & Clipped & Final round & At $\\hat r$ & $\\hat r$", rows)


TIMING_ORDER = [("fixed44", "Fixed (4,4)", "MLP"), ("tierfed", "\\tierfed{}", "MLP"),
                ("adacomm8", "AdaComm", "MLP"), ("lag_xi2000", "LAG, $\\xi=2000$", "MLP"),
                ("lag_xi5000", "LAG, $\\xi=5000$", "MLP"),
                ("fixed15_resnet", "Fixed (1,5)", "ResNet"), ("tierfed_resnet", "\\tierfed{}", "ResNet")]


def pperm(p):
    return "--" if p is None or not np.isfinite(p) else ("$<0.001$" if p < 1e-3 else f"{p:.3f}")


def tab_timing(tm):
    t = tm.set_index("name")
    rows, prev = [], None
    for n, lab, model in TIMING_ORDER:
        if n not in t.index:
            continue
        if prev is not None and model != prev:
            rows.append("\\midrule")
        prev = model
        r = t.loc[n]
        rows.append(" & ".join([lab, model, f4(r.auc_tx_rounds, 2), f4(r.auc_target_tx_rounds, 2),
                                f4(r.auc_target_mb, 2), f4(r.auc_total_mb, 2),
                                str(int(min(r.n_clean, r.n_shift))), pperm(r.p_perm)]) + "\\\\")
    return tabular("@{}llcccccc@{}", "Schedule & Model & Rounds with traffic & Target's rounds & Target's bytes "
                   "& All bytes & Runs per world & $p$", rows)


def family_of(r):
    if r.setting in ("tm_clean", "tm_shift"):
        return "Timing side channel, ResNet" if str(r["name"]).endswith("_resnet") else "Timing side channel, MLP"
    if r.eps == r.eps and r.eps is not None:
        return "Matched-$\\varepsilon$ DP ($q=1$)"
    if str(r.setting).endswith("_co"):
        return "Carry-over synchronisation step"
    if r.setting == "hospitals":
        return "Clinical centres"
    if r.setting == "resnet":
        return "ResNet backbone"
    if str(r.setting).startswith("a"):
        return "Label-skew sweep"
    if r["name"] in ("central", "hgb"):
        return "Reference bounds"
    if r["name"] in MAIN_ORDER:
        return "Main comparison"
    if r["name"] in ("tierfed_st_w8", "fixed44_st_w8", "tierfed_all"):
        return "Peak-aware variants"
    return "Accuracy--upload frontier"


FAMILY_DATA = [("Main comparison", "20 specialty groups"), ("Reference bounds", "pooled data, per-group test splits"),
               ("Accuracy--upload frontier", "20 specialty groups"), ("Peak-aware variants", "20 specialty groups"),
               ("Label-skew sweep", "Dirichlet $\\alpha\\in\\{0.1,0.3,1,10\\}$"),
               ("ResNet backbone", "20 specialty groups"), ("Clinical centres", "Fed-Heart-Disease, 4 centres"),
               ("Carry-over synchronisation step", "specialty groups, centres, ResNet"),
               ("Matched-$\\varepsilon$ DP ($q=1$)", "20 specialty groups"),
               ("Timing side channel, MLP", "clean or shifted target group"),
               ("Timing side channel, ResNet", "clean or shifted target group")]


def tab_families(df):
    d = df[df.setting != "derma"].copy(); d["family"] = d.apply(family_of, axis=1)
    rows = []; total = 0
    for fam, data in FAMILY_DATA:
        x = d[d.family == fam]
        if x.empty:
            continue
        per = x.groupby(["setting", "name"]).seed.nunique() if fam.startswith("Carry-over") else x.groupby("name").seed.nunique()
        seeds = f"{per.min()}" if per.min() == per.max() else f"{per.min()}--{per.max()}"
        nconf = x.groupby(["setting", "name"]).ngroups if fam in ("Label-skew sweep", "Carry-over synchronisation step") else x["name"].nunique()
        if fam.startswith("Timing"):
            nconf = x["name"].nunique(); seeds = seeds + " per world"
        rows.append(" & ".join([fam, data, str(nconf), seeds, str(len(x))]) + "\\\\")
        total += len(x)
    rows.append("\\midrule")
    rows.append(" & ".join(["Total", "", "", "", str(total)]) + "\\\\")
    d.attrs["total"] = total
    return tabular("@{}llccc@{}", "Family & Data and clients & Configurations & Seeds & Runs", rows), d


# --------------------------------------------------------------------------- #
def readings(history_csv: str) -> dict:
    """Test AUROC of one run under three ways of choosing what is reported."""
    h = pd.read_csv(history_csv)
    va = h["fed_val_auroc"].fillna(-np.inf).to_numpy(float); te = h["fed_auroc"].to_numpy(float)
    i = int(np.argmax(va))                                   # first maximum
    sm = np.array([va[max(0, j - 2): j + 1].mean() for j in range(len(va))])
    k = int(np.argmax(sm))
    return {"argmax": te[i], "smooth3": te[k], "last10": float(np.mean(te[-10:]))}


def selection_frame(base, gate):
    """TierFed minus FedPer and minus the best uniform period of the gate, per setting,
    under each selection rule, with the bootstrap of the main analysis."""
    rows = []
    for _, g in gate.iterrows():
        d = base[(base.setting == g.setting) & base.eps.isna()]
        runs = {n: d[d.name == n].set_index("seed")["file"] for n in ("tierfed", "fedper", g.best_uniform) if n}
        seeds = sorted(set.intersection(*[set(v.index) for v in runs.values()]))
        val = {n: {s: readings(str(v[s]).replace("_summary.json", "_history.csv")) for s in seeds} for n, v in runs.items()}
        for rule, _ in RULES:
            x = pd.DataFrame({"seed": seeds * 3,
                              "name": ["tierfed"] * len(seeds) + ["fedper"] * len(seeds) + ["uniform"] * len(seeds),
                              PRIMARY: [val["tierfed"][s][rule] for s in seeds] + [val["fedper"][s][rule] for s in seeds]
                              + [val[g.best_uniform][s][rule] for s in seeds]})
            a, b = paired(x, "tierfed", "fedper"), paired(x, "tierfed", "uniform")
            rows.append(dict(setting=g.setting, rule=rule, n=len(seeds), best=g.best_uniform,
                             d_fp=a["diff"], lo_fp=a["lo"], hi_fp=a["hi"], d_u=b["diff"], lo_u=b["lo"], hi_u=b["hi"]))
    return pd.DataFrame(rows)


def tab_selection(sel):
    rows, prev = [], None
    lab = dict(RULES)
    for _, r in sel.iterrows():
        if prev is not None and r.setting != prev:
            rows.append("\\midrule")
        first = r.setting != prev; prev = r.setting
        cell = f"{SETTING.get(r.setting, r.setting)}, {r.n} seeds, {k_label(r.best)}" if first else ""
        rows.append(" & ".join([cell, lab[r.rule], f"{fs(r.d_fp)} [{fs(r.lo_fp)}, {fs(r.hi_fp)}]",
                                f"{fs(r.d_u)} [{fs(r.lo_u)}, {fs(r.hi_u)}]"]) + "\\\\")
    return tabular("@{}llcc@{}", "Setting, seeds, best uniform period & Reported value "
                   "& \\tierfed{} $-$ FedPer [95\\% CI] & \\tierfed{} $-$ best uniform [95\\% CI]", rows)


# --------------------------------------------------------------------------- #
def planner_grid(w_slow: float, w_medium: float, rounds=60, warmup=5, k_max=8):
    """The privacy planner's choice on a grid of budgets, priors and quotas, computed
    with the class the experiments use."""
    from ..strategies.adaptive import PrivacyPlannedTierFed
    names = ["slow.w", "medium.w", "head.w"]
    sizes = {"slow.w": int(round(1e6 * w_slow)), "medium.w": int(round(1e6 * w_medium)), "head.w": 1}
    tiers = {"slow.w": "slow", "medium.w": "medium", "head.w": "fast"}
    rows = []
    # quotas below 0.19 are infeasible for the planner, which counts the warm-up: the sparsest
    # pair (8,8) already sends (5 + 6)/60 of the payload
    for eps, lam, rho in itertools.product((1.0, 2.0, 4.0, 8.0), (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0),
                                           (1.0, 0.4, 0.3, 0.25, 0.2)):
        p = PrivacyPlannedTierFed(names, sizes, tiers, rounds=rounds, warmup=warmup, k_max=k_max, rho=rho,
                                  lam=lam, epsilon=eps, q=1.0, delta=1e-5)
        rows.append(dict(eps=eps, lam=lam, rho=rho, ks=int(p.K["slow"]), km=int(p.K["medium"]),
                         releases=int(p.planned_releases), z=float(p.planned_sigma), payload=float(p.planned_payload)))
    return pd.DataFrame(rows)


def planner_eps_free(pg) -> bool:
    """The chosen pair is the same for every budget (the noise ratio of the objective
    does not depend on epsilon)."""
    return bool((pg.groupby(["rho", "lam"])[["ks", "km"]].nunique() == 1).all().all())


def tab_planner(pg):
    """One row per quota: the pair chosen for each prior weight (and each budget, when
    the choice depends on it)."""
    lams = sorted(pg.lam.unique())
    free = planner_eps_free(pg)
    rows = []
    for rho in sorted(pg.rho.unique(), reverse=True):
        for eps in ([pg.eps.min()] if free else sorted(pg.eps.unique())):
            x = pg[(pg.rho == rho) & (pg.eps == eps)].set_index("lam")
            cells = [f"{rho:g}"] + ([] if free else [f"{eps:g}"]) + [f"({x.loc[l, 'ks']},{x.loc[l, 'km']})" for l in lams]
            rows.append(" & ".join(cells) + "\\\\")
    head = "Quota $\\rho$ & " + ("" if free else "$\\varepsilon^\\star$ & ") + " & ".join(f"$\\lambda={l:g}$" for l in lams)
    return tabular("@{}c" + ("" if free else "c") + "c" * len(lams) + "@{}", head, rows)


# --------------------------------------------------------------------------- #
CO_ORDER = ["fixed44", "fixed88", "tierfed", "tierfed_db", "adacomm8"]
CO_NAMES = {"fixed44": "Fixed (4,4)", "fixed88": "Fixed (8,8)", "tierfed": "\\tierfed{}",
            "tierfed_db": "\\tierfed{}, rescaled dispersion", "adacomm8": "AdaComm"}
CO_SETTINGS = (("specialty", "Specialty groups"), ("hospitals", "Clinical centres"), ("resnet", "ResNet backbone"))


def sync_frame(base):
    """The round-sparse schedules under the two synchronisation steps (restart, the default
    of every other experiment, and carry-over), paired over the seeds both share."""
    rows = []
    for st, _ in CO_SETTINGS:
        a = base[(base.setting == st) & base.eps.isna()]
        b = base[(base.setting == st + "_co") & base.eps.isna()]
        for n in CO_ORDER:
            y = b[b.name == n].set_index("seed"); x = a[a.name == n].set_index("seed")
            if y.empty:
                continue
            seeds = sorted(set(y.index) & set(x.index)) if len(x) else sorted(y.index)
            row = dict(setting=st, name=n, n=len(seeds), has_restart=bool(len(x)))
            for tag, d in (("co", y), ("rs", x)):
                if d.empty:
                    continue
                d = d.loc[seeds]
                row.update({f"{tag}_auroc": d[PRIMARY].mean(), f"{tag}_round": d["valsel_round"].mean(),
                            f"{tag}_mb": d["upload_mb_valsel"].mean(), f"{tag}_total": d["total_upload_mb"].mean(),
                            f"{tag}_last": d["fed_auroc_last10_mean"].mean(), f"{tag}_final": d["fed_auroc_final"].mean()})
            if len(x) and len(seeds) >= 2:
                both = pd.concat([x.loc[seeds].reset_index().assign(name="rs"), y.loc[seeds].reset_index().assign(name="co")])
                r = paired(both, "co", "rs")
                row.update(d=r["diff"], lo=r["lo"], hi=r["hi"], p=r["p"])
                r = paired(both, "co", "rs", metric="fed_auroc_last10_mean")
                row.update(d_last=r["diff"], lo_last=r["lo"], hi_last=r["hi"], p_last=r["p"])
            rows.append(row)
    return pd.DataFrame(rows)


def sync_gate(base):
    """The matched-bytes contrast under the carry-over step: TierFed (and its rescaled variant)
    against the best uniform period among the carry-over runs and FedPer, which is the same
    run under both steps."""
    rows = []
    for st, _ in CO_SETTINGS:
        b = base[(base.setting == st + "_co") & base.eps.isna()]
        if "tierfed" not in set(b.name):
            continue
        fp_ = base[(base.setting == st) & base.eps.isna() & (base.name == "fedper")]
        d = pd.concat([b[b.name != "fedper"], fp_])
        for n in ("tierfed", "tierfed_db"):
            t = d[d.name == n]
            if t.empty:
                continue
            bu = best_uniform(d, MATCH_TOL * t.total_upload_mb.mean())
            r = paired(d, n, bu) if bu else {"diff": np.nan, "lo": np.nan, "hi": np.nan, "p": np.nan, "n": 0}
            k = t[["sch_K_slow", "sch_K_medium"]].dropna().round().astype(int)
            mode = k.value_counts().index[0] if len(k) else (0, 0)
            rows.append(dict(setting=st, name=n, best=bu, same=bool(bu) and identical(d, n, bu),
                             same44=identical(d, n, "fixed44"), d=r["diff"], lo=r["lo"],
                             hi=r["hi"], p=r["p"], n=r.get("n", 0), tf_mb=t.total_upload_mb.mean(),
                             u_mb=d[d.name == bu].total_upload_mb.mean() if bu else np.nan,
                             k_end=f"({mode[0]},{mode[1]})", k_early=early_k(t),
                             diff_rate=t["sch_frac_rounds_tier_differentiated"].mean()
                             if "sch_frac_rounds_tier_differentiated" in t else np.nan,
                             tf_auroc=t[PRIMARY].mean()))
    return pd.DataFrame(rows)


def tab_sync(sf):
    lab = dict(CO_SETTINGS)
    rows, prev = [], None
    for _, r in sf.iterrows():
        if prev is not None and r.setting != prev:
            rows.append("\\midrule")
        first = r.setting != prev; prev = r.setting
        rs = ([f4(r.rs_auroc), f4(r.rs_round, 0), f4(r.rs_mb, 1), f4(r.rs_last)] if r.has_restart else ["--"] * 4)
        diff = (f"{fs(r.d)} [{fs(r.lo)}, {fs(r.hi)}]" if r.has_restart and np.isfinite(r.get("d", np.nan)) else "--")
        rows.append(" & ".join([f"{lab[r.setting]} ({r.n})" if first else "", CO_NAMES[r["name"]]] + rs
                               + [f4(r.co_auroc), f4(r.co_round, 0), f4(r.co_mb, 1), f4(r.co_last), diff,
                                  fp(r.p) if r.has_restart and np.isfinite(r.get("p", np.nan)) else "--"]) + "\\\\")
    head = (" & & \\multicolumn{4}{c}{Restart step} & \\multicolumn{4}{c}{Carry-over step} & & \\\\\n"
            "\\cmidrule(lr){3-6}\\cmidrule(lr){7-10}\n"
            "Setting (seeds) & Schedule & AUROC at $\\hat r$ & $\\hat r$ & MB to $\\hat r$ & Last 10 "
            "& AUROC at $\\hat r$ & $\\hat r$ & MB to $\\hat r$ & Last 10 & Difference at $\\hat r$ [95\\% CI] & $p$")
    return tabular("@{}llcccccccccc@{}", head, rows)


def dispersion_frame(results, dataset="diabetes", model="mlp", clients=20):
    """Mean dispersion of the uploaded slices (the measured proxy) in the synchronisation
    rounds after warm-up, per uniform period and synchronisation step."""
    import glob as _glob
    rows = []
    L = Path(results) / "logs"
    for step, pre in (("restart", ""), ("carry", "co_")):
        for k, strat, name in ((1, "fedper", "fedper"), (2, "fixed", "fixed22"), (4, "fixed", "fixed44"),
                               (6, "fixed", "fixed66"), (8, "fixed", "fixed88")):
            if step == "carry" and k == 1:
                continue
            fsn = sorted(_glob.glob(str(L / f"{dataset}_{model}_{strat}_R60_C{clients}_s*_au_{pre}{name}_history.csv")))
            v = []
            for f in fsn:
                h = pd.read_csv(f)
                tx = h[(h["round"] > 5) & (h["payload_fraction"] > 0)]
                if len(tx) and "divergence_slow" in tx:
                    v.append((tx["divergence_slow"].mean(), tx["divergence_medium"].mean()))
            if v:
                v = np.array(v)
                rows.append(dict(step=step, k=k, n=len(v), slow=float(v[:, 0].mean()), medium=float(v[:, 1].mean())))
    return pd.DataFrame(rows)


def sci(x):
    return "--" if x is None or not np.isfinite(x) else f"{x * 1e3:.2f}"


def tab_dispersion(dd):
    rows = []
    for k in sorted(dd.k.unique()):
        a = dd[(dd.k == k) & (dd.step == "restart")]; b = dd[(dd.k == k) & (dd.step == "carry")]
        g = lambda x, c: sci(float(x[c].iloc[0])) if len(x) else "--"   # noqa: E731
        rows.append(" & ".join([str(int(k)), str(int(a.n.iloc[0])) if len(a) else "--", g(a, "slow"), g(a, "medium"),
                                str(int(b.n.iloc[0])) if len(b) else "--", g(b, "slow"), g(b, "medium")]) + "\\\\")
    head = (" & \\multicolumn{3}{c}{Restart step} & \\multicolumn{3}{c}{Carry-over step}\\\\\n"
            "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n"
            "Period $K$ & Seeds & Slow tier & Medium tier & Seeds & Slow tier & Medium tier")
    return tabular("@{}ccccccc@{}", head, rows)


# --------------------------------------------------------------------------- #
def site_weights(dataset: str, n_clients: int, seeds, **kw):
    """Aggregation weights of the training splits: sum of squares and largest share."""
    from ..data import load_dataset
    out = []
    for s in seeds:
        ds = load_dataset(dataset, n_clients=n_clients, seed=s, **kw)
        tr, _, _ = ds.client_val_test_split(frac=0.3, val_share=1 / 3, seed=s)
        n = np.array([len(i) for i in tr], float); p = n / n.sum()
        out.append(dict(seed=s, sum_p2=float((p ** 2).sum()), n_sites=len(n), sizes=n.astype(int).tolist(),
                        total=[len(i) for i in ds.client_indices]))
    return out


def numbers(df, base, gate, tm, sel, pg, W, results, n_runs=0):
    sp = base[(base.setting == "specialty") & base.eps.isna()]
    mt = main_table(base)
    # ---- upload in the two accountings (reviewer: say which one the headline uses)
    for n, key in (("tierfed", "Tierfed"), ("fedper", "Fedper"), ("adacomm8", "Adacomm"), ("fixed88", "FixedEight")):
        if n in mt.index:
            W.num(f"tm{key}MbSel", f4(mt.loc[n, "upload_mb_valsel_mean"], 1))
            W.num(f"tm{key}MbAll", f4(mt.loc[n, "total_upload_mb_mean"], 1))
    a, b = mt.loc["tierfed"], mt.loc["fedper"]
    W.num("tmSaveSelPct", f4(100 * (1 - a["upload_mb_valsel_mean"] / b["upload_mb_valsel_mean"]), 0))
    W.num("tmSaveAllPct", f4(100 * (1 - a["total_upload_mb_mean"] / b["total_upload_mb_mean"]), 0))
    W.num("tmAbsTierfedDFedper", f4(abs(a["d_fedper_valsel"])))
    W.num("tmTierfedRound", f4(a["valsel_round_mean"], 0)); W.num("tmFedperRound", f4(b["valsel_round_mean"], 0))
    W.flag("tmTierfedIsFixed", identical(sp, "tierfed", "fixed44"))
    W.flag("tmStaggerIsFixed", identical(sp, "tierfed_st", "fixed44_st"))
    # ---- seeds per family
    W.num("tmMainSeeds", str(sp[sp.name == "tierfed"].seed.nunique()))
    rn = base[(base.setting == "resnet") & base.eps.isna()]
    W.num("tmResnetSeeds", str(rn[rn.name == "tierfed"].seed.nunique()))
    het = base[base.setting.str.match(r"^a[0-9.]+$") & base.eps.isna()]
    if len(het):
        W.num("tmHetSeeds", str(int(het[het.name == "tierfed"].groupby("setting").seed.nunique().min())))
    W.num("tmRuns", f"{int(n_runs):,}")
    # ---- gate: Holm over the settings in which the runs are not identical, and the rule's outcome
    live = [r for _, r in gate.iterrows()
            if r.best_uniform and not identical(base[(base.setting == r.setting) & base.eps.isna()], "tierfed", r.best_uniform)]
    adj = holm([r.p_vs_uniform for r in live]) if live else []
    for r, ph in zip(live, adj):
        W.num("tmGate" + _safe(r.setting.replace(".", "p").title()) + "PHolm", fp(float(ph)).replace("$", ""))
    W.flag("tmGateNoWin", not any(np.isfinite(r.gain_lo) and r.gain_lo > 0 and r.gain_vs_uniform >= PRACTICAL
                                   for _, r in gate.iterrows()))
    W.flag("tmGateNoHolmWin", not any(float(ph) < 0.05 and r.gain_vs_uniform >= PRACTICAL for r, ph in zip(live, adj)))
    W.num("tmGateSettings", str(len(gate)))
    W.num("tmGateIdentical", str(len(gate) - len(live)))
    psis = {}
    for _, r in gate.iterrows():
        t = base[(base.setting == r.setting) & base.eps.isna() & (base.name == "tierfed")]
        psis[r.setting] = float(t["psi"].mean()) if "psi" in t else np.nan
    small = {k: v for k, v in psis.items() if np.isfinite(v) and v < 0.05}
    W.num("tmGateSmallPsi", str(len(small)))
    ident = [k for k in small if identical(base[(base.setting == k) & base.eps.isna()], "tierfed", "fixed44")]
    W.num("tmGateSmallIdentical", str(len(ident)))
    W.num("tmGateSettingsWord", word(len(gate))); W.num("tmGateSmallPsiWord", word(len(small)))
    W.num("tmGateSmallIdenticalWord", word(len(ident)))
    kmode = {k: early_k(base[(base.setting == k) & base.eps.isna() & (base.name == "tierfed")]) for k in small}
    W.flag("tmGateSmallAllFourFour", all(v == "(4,4)" for v in kmode.values()))
    if small:
        W.num("tmGatePsiSmallMax", f4(100 * max(small.values()), 1))
    W.flag("tmGateOnlyResnetLarge", set(psis) - set(small) == {"resnet"})
    def _win(r):
        return bool(np.isfinite(r.gain_lo) and r.gain_lo > 0 and r.gain_vs_uniform >= PRACTICAL)
    W.flag("tmHetNoWin", not any(_win(r) for _, r in gate.iterrows() if re.match(r"^a[0-9.]+$", r.setting)))
    W.flag("tmResnetNoWin", not any(_win(r) for _, r in gate.iterrows() if r.setting == "resnet"))
    five = [fam for fam in ("het", "frontier") if True]
    het_seeds = base[base.setting.str.match(r"^a[0-9.]+$") & base.eps.isna() & (base.name == "tierfed")].groupby("setting").seed.nunique()
    fr = sp[sp.name.isin(["fixed22", "fixed66", "adacomm4", "adacomm16", "tierfed_rho0.15", "tierfed_rho0.4"])].groupby("name").seed.nunique()
    W.flag("tmFiveSeedFamilies", bool(len(het_seeds) and (het_seeds == 5).all() and len(fr) and (fr == 5).all()))
    pkv = sp[sp.name.isin(["fixed44_st_w8", "tierfed_st_w8"])].groupby("name").seed.nunique()
    if len(pkv):
        W.num("tmPeakSeeds", str(int(pkv.min())))
    # ---- reference bounds on the frontier's seeds
    fs_seeds = common_seeds(sp[~sp.name.isin(["central", "hgb"])])
    for n, key in (("central", "Central"), ("hgb", "Hgb")):
        v = sp[(sp.name == n) & sp.seed.isin(fs_seeds)][PRIMARY]
        if len(v):
            W.num(f"tm{key}Frontier", f4(v.mean()))
    W.num("tmFrontierSeeds", str(len(fs_seeds)))
    # ---- selection rules
    if sel is not None and len(sel):
        for _, r in sel.iterrows():
            key = "tmSel" + _safe(r.setting.replace(".", "p").title()) + _safe(r.rule.title())
            W.num(key + "Fp", fs(r.d_fp)); W.num(key + "FpLo", fs(r.lo_fp)); W.num(key + "FpHi", fs(r.hi_fp))
            W.num(key + "U", fs(r.d_u)); W.num(key + "ULo", fs(r.lo_u)); W.num(key + "UHi", fs(r.hi_u))
        W.flag("tmSelNoWin", not bool(((sel.lo_u > 0) & (sel.d_u >= PRACTICAL)).any()))
        behind = sel[(sel.hi_fp < 0) & (sel.d_fp <= -PRACTICAL)]
        W.num("tmSelBehindFedper", str(len(behind)))
        ahead = sel[(sel.lo_fp > 0) & (sel.rule == "last10")]
        W.num("tmSelLastAhead", str(len(ahead)))
        sb = behind[behind.rule == "smooth3"]
        W.num("tmSelSmoothBehind", str(len(sb))); W.num("tmSelSmoothBehindWord", word(len(sb)))
        W.num("tmSelSmoothBehindList", "; ".join(
            f"{SETTING.get(r.setting, r.setting)}: {fs(r.d_fp)} [{fs(r.lo_fp)}, {fs(r.hi_fp)}]" for r in sb.itertuples()) or "none")
        W.flag("tmSelSmoothBehindTen", bool(len(behind[(behind.setting == "a10") & (behind.rule == "smooth3")])))
        W.flag("tmSelSmoothBehindResnet", bool(len(behind[(behind.setting == "resnet") & (behind.rule == "smooth3")])))
    # ---- matched privacy, paired with local training on the same seeds
    d, loc, seeds = dp_frames(df)
    if len(d):
        rel = d[d.name != "oneshot"]
        fin = d.groupby("name")["fed_auroc_final"].mean(); selv = d.groupby("name")[PRIMARY].mean()
        W.num("tmDpSeeds", str(len(seeds)))
        W.num("tmDpFinalMin", f4(fin.min())); W.num("tmDpFinalMax", f4(fin.max()))
        W.num("tmDpLocalFinal", f4(loc["fed_auroc_final"].mean())); W.num("tmDpLocalSel", f4(loc[PRIMARY].mean()))
        lf = loc.set_index("seed")["fed_auroc_final"]
        below = sum(float(r.fed_auroc_final) < float(lf[r.seed]) for r in d.itertuples())
        W.num("tmDpBelow", str(int(below))); W.num("tmDpPairs", str(len(d)))
        W.flag("tmDpAllMeansBelowLocal", bool((fin < loc["fed_auroc_final"].mean()).all()))
        W.num("tmDpBestSel", f4(selv[[n for n in selv.index if n != "oneshot"]].max()))
        W.num("tmDpOneshotSel", f4(selv.get("oneshot", np.nan))); W.num("tmDpOneshotFinal", f4(fin.get("oneshot", np.nan)))
        one = d[d.name == "oneshot"]
        if len(one):
            W.num("tmDpOneshotRound", f4(one["valsel_round"].mean(), 0))
            W.flag("tmDpOneshotSelBeforeRelease", bool((one["valsel_round"] < 60).all()))
        W.num("tmDpFedperClip", f4(d[d.name == "fedper"]["clip_fraction"].mean(), 2))
        if "tierfed_dp" in set(d.name):
            k = d[d.name == "tierfed_dp"][["sch_K_slow", "sch_K_medium"]].dropna().astype(int)
            mode = k.value_counts().index[0]
            W.num("tmDpPlannerK", f"({mode[0]},{mode[1]})")
            W.flag("tmDpPlannerSameK", bool((k.nunique() == 1).all()))
        curves = []
        for n in ("fedper", "fixed44", "fixed88", "tierfed_dp"):
            hs = [pd.read_csv(str(f).replace("_summary.json", "_history.csv"))["fed_auroc"].to_numpy(float)
                  for f in d[d.name == n]["file"]]
            if hs:
                m = min(len(h) for h in hs); curves.append(np.mean([h[:m] for h in hs], 0).max())
        if curves:
            W.num("tmDpCurveMax", f4(max(curves)))
    # ---- 100 sites, final-round protocol (supplement table)
    hund = sorted(Path(results, "logs").glob("diabetes_mlp_*_R60_C100_s*_dpe*_summary.json"))
    vals = {}
    for f in hund:
        dd = json.loads(f.read_text()); cfg = dd.get("config", {})
        vals.setdefault((cfg.get("strategy"), cfg.get("tag")), []).append(dd["summary"]["fed_auroc_final"])
    if vals:
        m = [float(np.mean(v)) for v in vals.values()]
        W.num("tmHundredMin", f4(min(m))); W.num("tmHundredMax", f4(max(m))); W.num("tmHundredArms", str(len(m)))
    # ---- privacy planner on a grid
    if pg is not None and len(pg):
        cur = pg[(pg.eps == 4.0) & (pg.lam == 1.0) & (pg.rho == 1.0)].iloc[0]
        W.num("tmPlanK", f"({cur.ks},{cur.km})"); W.num("tmPlanGrid", str(len(pg)))
        eq = pg[pg.ks == pg.km]; ne = pg[pg.ks != pg.km]
        W.num("tmPlanEqual", str(len(eq))); W.num("tmPlanUnequal", str(len(ne)))
        W.flag("tmPlanAlwaysEqual", len(ne) == 0)
        if len(ne):
            pairs = sorted({(int(r.ks), int(r.km)) for r in ne.itertuples()})
            W.num("tmPlanUnequalPairs", ", ".join(f"({a_},{b_})" for a_, b_ in pairs))
            W.num("tmPlanUnequalRho", ", ".join(f"{x:g}" for x in sorted(ne.rho.unique())))
            W.flag("tmPlanUnequalOnlyAtBindingQuota", bool((ne.rho < 1.0).all()))
        W.flag("tmPlanEqualWithoutQuota", bool((pg[pg.rho == 1.0].ks == pg[pg.rho == 1.0].km).all()))
        W.flag("tmPlanEpsFree", planner_eps_free(pg))
        W.num("tmPlanLambdas", str(pg.lam.nunique())); W.num("tmPlanQuotas", str(pg.rho.nunique()))
        cells = pg.drop_duplicates(["rho", "lam"])
        W.num("tmPlanCells", str(len(cells))); W.num("tmPlanCellsEqual", str(int((cells.ks == cells.km).sum())))
        W.num("tmPlanCellsUnequal", str(int((cells.ks != cells.km).sum())))
    # ---- timing side channel
    t = tm.set_index("name") if tm is not None and len(tm) else pd.DataFrame()
    leaks_any = False
    for n, _, _ in TIMING_ORDER:
        if n not in t.index:
            continue
        r = t.loc[n]; key = "tmT" + _safe(n.title().replace("_", ""))
        aucs = [r.auc_tx_rounds, r.auc_target_tx_rounds, r.auc_target_mb, r.auc_total_mb]
        W.num(key + "AucAll", f4(r.auc_total_mb, 2)); W.num(key + "AucTarget", f4(r.auc_target_tx_rounds, 2))
        W.num(key + "AucMax", f4(float(np.nanmax(aucs)), 2)); W.num(key + "AucMin", f4(float(np.nanmin(aucs)), 2))
        W.num(key + "P", pperm(r.p_perm).replace("$", "")); W.num(key + "Runs", str(int(min(r.n_clean, r.n_shift))))
        W.num(key + "RunsWord", word(int(min(r.n_clean, r.n_shift))))
        pe = pperm(r.p_perm).replace("$", "")
        W.num(key + "PEq", pe if pe.startswith("<") else "=" + pe)
        obs = ("the number of rounds with traffic", "the number of rounds in which the target transmits",
               "the upload of the target", "the total upload of all clients")
        best = int(np.nanargmax(np.abs(np.array(aucs) - 0.5)))
        W.num(key + "BestObs", obs[best]); W.num(key + "BestAuc", f4(aucs[best], 2))
        leaks_any = leaks_any or bool(np.nanmax(np.abs(np.array(aucs) - 0.5)) >= 0.25 and r.p_perm < 0.05)
        W.num(key + "OtherSkipPct", f4(100 * r.other_skip_share, 0))
        W.flag(key + "Silent", bool(all(abs(x - 0.5) < 1e-12 for x in aucs)))
        W.flag(key + "TargetNeverSkips", bool(r.target_skips == 0))
        W.flag(key + "Leaks", bool(np.nanmax(np.abs(np.array(aucs) - 0.5)) >= 0.25 and r.p_perm < 0.05))
    W.flag("tmTAnyLeak", bool(leaks_any))
    # ---- main benchmark: aggregation weights
    try:
        sw = site_weights("diabetes", 20, sorted(sp.seed.unique()))
        s2 = float(np.mean([x["sum_p2"] for x in sw]))
        W.num("tmSumPSq", f4(s2, 3)); W.num("tmNeff", f4(1 / s2, 1))
    except Exception as e:                      # data not on this machine: the macro stays undefined
        print("site weights skipped:", e)
    # ---- real hospitals
    hp = base[(base.setting == "hospitals") & base.eps.isna()]
    if len(hp):
        mh = main_table(base, "hospitals", HOSP_ORDER)
        W.num("tmHospSeeds", str(hp[hp.name == "tierfed"].seed.nunique()))
        for n, key in (("tierfed", "Tierfed"), ("fedper", "Fedper"), ("fedavg", "Fedavg"), ("local", "Local"),
                       ("adacomm8", "Adacomm"), ("fixed44", "FixedFour"), ("fixed88", "FixedEight"),
                       ("fixed22", "FixedTwo"), ("fedprox", "Fedprox"), ("fedlama", "Fedlama"), ("lag", "Lag"),
                       ("fixed44_st", "FixedFourSt"), ("tierfed_st", "TierfedSt")):
            if n in mh.index:
                r = mh.loc[n]
                W.num(f"tmHosp{key}Auroc", f4(r[PRIMARY + "_mean"])); W.num(f"tmHosp{key}Last", f4(r["fed_auroc_last10_mean_mean"]))
                W.num(f"tmHosp{key}MbSel", f4(r["upload_mb_valsel_mean"], 1)); W.num(f"tmHosp{key}MbAll", f4(r["total_upload_mb_mean"], 1))
                W.num(f"tmHosp{key}Round", f4(r["valsel_round_mean"], 0))
                if n != "fedper":
                    W.num(f"tmHosp{key}DFedper", fs(r["d_fedper_valsel"])); W.num(f"tmHosp{key}DFedperP", fp(r["p_fedper_valsel"]).replace("$", ""))
                    W.num(f"tmHosp{key}DFedperLo", fs(r["ci_lo_fedper_valsel"])); W.num(f"tmHosp{key}DFedperHi", fs(r["ci_hi_fedper_valsel"]))
        full = [n for n in ("fedavg", "fedprox", "fedper", "fedlama", "lag", "fixed22") if n in mh.index]
        W.num("tmHospFullMin", f4(mh.loc[full, PRIMARY + "_mean"].min())); W.num("tmHospFullMax", f4(mh.loc[full, PRIMARY + "_mean"].max()))
        W.num("tmHospWorstMin", f4(mh["fed_auroc_site_min_valsel_mean"].min(), 2)); W.num("tmHospWorstMax", f4(mh["fed_auroc_site_min_valsel_mean"].max(), 2))
        half = [np.diff(ci(hp[hp.name == n][PRIMARY]))[0] / 2 for n in mh.index]
        W.num("tmHospCiHalf", f4(float(np.mean(half)), 2))
        W.num("tmHospAbsTierfedDFedper", f4(abs(mh.loc["tierfed", "d_fedper_valsel"])))
        r = mh.loc["tierfed"]
        W.num("tmHospNiTenP", fp(r["p_ni010_fedper_valsel"]).replace("$", "")); W.num("tmHospNiFiveP", fp(r["p_ni005_fedper_valsel"]).replace("$", ""))
        W.flag("tmHospNiTen", bool(r["p_ni010_fedper_valsel"] < 0.05))
        tfh = hp[hp.name == "tierfed"]
        W.num("tmHospPsi", f4(100 * tfh.psi.mean(), 1)); W.num("tmHospPsiMax", f4(100 * tfh.psi.max(), 1))
        k = tfh[["sch_K_slow", "sch_K_medium"]].dropna().astype(int)
        W.flag("tmHospAlwaysFourFour", bool(((k.sch_K_slow == 4) & (k.sch_K_medium == 4)).all()))
        W.flag("tmHospTierfedIsFixed", identical(hp, "tierfed", "fixed44"))
        sc = hp["n_clients_single_class_valsel"].dropna()
        W.num("tmHospSingleRuns", f4(100 * float((sc > 0).mean()), 0))
        d_l = paired(hp, "fedavg", "local"); W.num("tmHospFedavgVsLocal", fs(d_l["diff"]))
        W.num("tmHospFedavgVsLocalLo", fs(d_l["lo"])); W.num("tmHospFedavgVsLocalHi", fs(d_l["hi"])); W.num("tmHospFedavgVsLocalP", fp(d_l["p"]).replace("$", ""))
        d_l = paired(hp, "tierfed", "local"); W.num("tmHospTierfedVsLocal", fs(d_l["diff"]))
        W.num("tmHospTierfedVsLocalLo", fs(d_l["lo"])); W.num("tmHospTierfedVsLocalHi", fs(d_l["hi"])); W.num("tmHospTierfedVsLocalP", fp(d_l["p"]).replace("$", ""))
        g = gate[gate.setting == "hospitals"]
        if len(g):
            W.num("tmHospBest", k_label(g.iloc[0].best_uniform))
        fed = [n for n in HOSP_ORDER if n in mh.index and n not in ("local", "fedper")]
        W.num("tmHospSpread", f4(mh.loc[fed, "d_fedper_valsel"].abs().max()))
        W.num("tmHospAurocMin", f4(mh.loc[[n for n in HOSP_ORDER if n in mh.index and n != "local"], PRIMARY + "_mean"].min()))
        W.num("tmHospAurocMax", f4(mh.loc[[n for n in HOSP_ORDER if n in mh.index and n != "local"], PRIMARY + "_mean"].max()))
        f0 = sorted(hp["file"])[0]; ds = json.loads(Path(f0).read_text())["dataset"]
        W.num("tmHospPatients", str(int(ds["n_train"]) + int(ds["n_test"]))); W.num("tmHospAtSites", str(int(ds["n_train"])))
        W.num("tmHospFeatures", str(int(ds["n_features"]))); W.num("tmHospPrevalence", f4(100 * ds["prevalence_train"], 0))
        tr = json.loads(Path(sorted(hp[hp.name == "tierfed"]["file"])[0]).read_text()).get("tier_report") or {}
        if tr:
            W.num("tmHospParams", f"{int(tr['total']['params']):,}")
            W.num("tmHospTierSlowPct", f4(tr["slow"]["pct"], 1)); W.num("tmHospTierMediumPct", f4(tr["medium"]["pct"], 1))
            W.num("tmHospTierFastPct", f4(tr["fast"]["pct"], 1))
        pr = ds.get("partition_report") or {}
        W.num("tmHospPrevMin", f4(100 * pr.get("prevalence_min", np.nan), 0)); W.num("tmHospPrevMax", f4(100 * pr.get("prevalence_max", np.nan), 0))
        W.num("tmHospSizeMin", str(int(pr.get("size_min", 0)))); W.num("tmHospSizeMax", str(int(pr.get("size_max", 0))))
        try:
            sw = site_weights("heart", 0, sorted(hp.seed.unique()))
            W.num("tmHospNeff", f4(1 / float(np.mean([x["sum_p2"] for x in sw])), 1))
        except Exception as e:
            print("hospital weights skipped:", e)
    # ---- staggering, in numbers the text quotes
    for n, key in (("fixed44", "Aligned"), ("fixed44_st", "Staggered")):
        v = sp[sp.name == n]
        if len(v):
            W.num(f"tm{key}Mb", f4(v["total_upload_mb"].mean(), 1)); W.num(f"tm{key}Energy", f4(v["net_nr_cell_edge_energy_j"].mean(), 1))
            W.num(f"tm{key}Wake", f4(v["wakeups"].mean(), 0))
    return mt


def sync_vs_fedper(base, W):
    key = {"specialty": "Spec", "hospitals": "Hosp", "resnet": "Res"}
    for st, _ in CO_SETTINGS:
        b = base[(base.setting == st + "_co") & base.eps.isna()]
        a = base[(base.setting == st) & base.eps.isna() & (base.name == "fedper")]
        if b.empty or a.empty:
            continue
        d = pd.concat([b, a])
        for n, nk in (("fixed44", "FixedFour"), ("fixed88", "FixedEight"), ("adacomm8", "Adacomm"), ("tierfed", "Tierfed")):
            if n in set(b.name):
                r = paired(d, n, "fedper")
                k = "tmSyncFp" + key[st] + nk
                W.num(k + "D", fs(r["diff"])); W.num(k + "Lo", fs(r["lo"])); W.num(k + "Hi", fs(r["hi"]))
                W.num(k + "P", fp(r["p"]).replace("$", ""))
        W.num("tmSyncFp" + key[st] + "FedperMb", f4(a["upload_mb_valsel"].mean(), 1))
        W.num("tmSyncFp" + key[st] + "FedperRound", f4(a["valsel_round"].mean(), 0))


def sync_resnet(base, W):
    b = base[(base.setting == "resnet_co") & base.eps.isna()]
    if not {"tierfed", "fixed44"} <= set(b.name):
        return
    ok = True
    for n, nk in (("tierfed", "Tierfed"), ("tierfed_db", "TierfedDb")):
        if n not in set(b.name):
            continue
        r = paired(b, n, "fixed44")
        k = "tmSyncResVsFour" + nk
        W.num(k + "D", fs(r["diff"])); W.num(k + "Lo", fs(r["lo"])); W.num(k + "Hi", fs(r["hi"]))
        W.num(k + "P", fp(r["p"]).replace("$", "")); W.num(k + "N", str(int(r["n"])))
        ok &= not (np.isfinite(r["lo"]) and r["lo"] > 0 and r["diff"] >= PRACTICAL)
        W.flag(k + "CiAboveZero", bool(np.isfinite(r["lo"]) and r["lo"] > 0))
        W.flag(k + "BelowThreshold", bool(abs(r["diff"]) < PRACTICAL))
        W.num(k + "Abs", f4(abs(r["diff"])))
        t = b[b.name == n]
        W.num(k + "Mb", f4(t.total_upload_mb.mean(), 0)); W.num(k + "KEarly", early_k(t))
        W.num(k + "DiffRate", f4(t["sch_frac_rounds_tier_differentiated"].mean(), 2))
        W.flag(k + "Splits", bool(t["sch_frac_rounds_tier_differentiated"].mean() > 0.5))
    W.num("tmSyncResFourMb", f4(b[b.name == "fixed44"].total_upload_mb.mean(), 0))
    W.num("tmSyncResSeeds", str(b[b.name == "tierfed"].seed.nunique()))
    W.num("tmSyncResSeedsWord", word(b[b.name == "tierfed"].seed.nunique()))
    W.flag("tmSyncResNoGainVsFour", bool(ok))


def sync_numbers(sf, sg, dd, W):
    key = {"specialty": "Spec", "hospitals": "Hosp", "resnet": "Res"}
    nm = {"fixed44": "FixedFour", "fixed88": "FixedEight", "tierfed": "Tierfed", "tierfed_db": "TierfedDb",
          "adacomm8": "Adacomm"}
    for _, r in sf.iterrows():
        k = "tmSync" + key[r.setting] + nm[r["name"]]
        W.num(k + "Seeds", str(int(r.n)))
        W.num(k + "CoAuroc", f4(r.co_auroc)); W.num(k + "CoRound", f4(r.co_round, 0)); W.num(k + "CoMb", f4(r.co_mb, 1))
        W.num(k + "CoLast", f4(r.co_last)); W.num(k + "CoTotal", f4(r.co_total, 1))
        if r.has_restart:
            W.num(k + "RsAuroc", f4(r.rs_auroc)); W.num(k + "RsRound", f4(r.rs_round, 0)); W.num(k + "RsMb", f4(r.rs_mb, 1))
            W.num(k + "RsLast", f4(r.rs_last))
            if np.isfinite(r.get("d", np.nan)):
                W.num(k + "D", fs(r.d)); W.num(k + "Lo", fs(r.lo)); W.num(k + "Hi", fs(r.hi)); W.num(k + "P", fp(r.p).replace("$", ""))
                W.num(k + "DLast", fs(r.d_last)); W.num(k + "LoLast", fs(r.lo_last)); W.num(k + "HiLast", fs(r.hi_last))
                W.flag(k + "Relevant", bool(abs(r.d) >= PRACTICAL and (r.lo > 0 or r.hi < 0)))
                W.flag(k + "LastLower", bool(r.hi_last < 0)); W.flag(k + "EarlierRound", bool(r.co_round < r.rs_round))
    if len(sf):
        x = sf[sf.has_restart & sf.d.notna()] if "d" in sf else sf.iloc[0:0]
        if len(x):
            W.num("tmSyncMaxAbsD", f4(float(x.d.abs().max())))
            W.flag("tmSyncNoneRelevant", bool(not ((x.d.abs() >= PRACTICAL) & ((x.lo > 0) | (x.hi < 0))).any()))
            W.flag("tmSyncAllEarlier", bool((x.co_round < x.rs_round).all()))
            W.flag("tmSyncAllLastLower", bool((x.hi_last < 0).all()))
            for st_, kk in (("specialty", "Spec"), ("hospitals", "Hosp"), ("resnet", "Res")):
                xx = x[x.setting == st_]
                if len(xx):
                    W.flag(f"tmSync{kk}NoneRelevant", bool(not ((xx.d.abs() >= PRACTICAL) & ((xx.lo > 0) | (xx.hi < 0))).any()))
                    W.flag(f"tmSync{kk}AllHigher", bool((xx.d > 0).all())); W.flag(f"tmSync{kk}AllLower", bool((xx.d < 0).all()))
                    W.num(f"tmSync{kk}DMin", fs(float(xx.d.min()))); W.num(f"tmSync{kk}DMax", fs(float(xx.d.max())))
            xs = x[x.setting == "specialty"]
            if len(xs):
                W.num("tmSyncSpecLastMin", f4(float(xs.d_last.abs().min()))); W.num("tmSyncSpecLastMax", f4(float(xs.d_last.abs().max())))
                W.flag("tmSyncSpecAllLastLower", bool((xs.hi_last < 0).all()))
                W.flag("tmSyncSpecAllEarlier", bool((xs.co_round < xs.rs_round).all()))
    for _, r in sg.iterrows():
        k = "tmSyncGate" + key[r.setting] + nm[r["name"]]
        W.num(k + "Best", k_label(r.best) if r.best else "--"); W.flag(k + "Identical", bool(r.same))
        W.flag(k + "IsFixedFour", bool(r.same44))
        W.num(k + "D", fs(r.d)); W.num(k + "Lo", fs(r.lo)); W.num(k + "Hi", fs(r.hi)); W.num(k + "P", fp(r.p).replace("$", ""))
        W.num(k + "KEnd", r.k_end); W.num(k + "KEarly", r.k_early); W.num(k + "DiffRate", f4(r.diff_rate, 2))
        W.num(k + "Mb", f4(r.tf_mb, 1)); W.num(k + "UMb", f4(r.u_mb, 1)); W.num(k + "Auroc", f4(r.tf_auroc))
        W.flag(k + "Wins", bool(np.isfinite(r.lo) and r.lo > 0 and r.d >= PRACTICAL))
    if len(sg):
        W.flag("tmSyncGateNoWin", bool(not any(np.isfinite(r.lo) and r.lo > 0 and r.d >= PRACTICAL for r in sg.itertuples())))
    if len(dd):
        rs = dd[dd.step == "restart"]; co = dd[dd.step == "carry"]
        if len(rs):
            W.num("tmDispRsSlowMin", sci(rs.slow.min())); W.num("tmDispRsSlowMax", sci(rs.slow.max()))
            W.num("tmDispRsMedMin", sci(rs.medium.min())); W.num("tmDispRsMedMax", sci(rs.medium.max()))
            W.num("tmDispRsSpread", f4(max(rs.slow.max() / rs.slow.min(), rs.medium.max() / rs.medium.min()), 2))
        for k in (4, 8):
            a = rs[rs.k == k]; b = co[co.k == k]
            if len(a) and len(b):
                kk = {4: "Four", 8: "Eight"}[k]
                W.num(f"tmDispRatioSlow{kk}", f4(float(b.slow.iloc[0] / a.slow.iloc[0]), 1))
                W.num(f"tmDispRatioMed{kk}", f4(float(b.medium.iloc[0] / a.medium.iloc[0]), 1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="paper/tmlcn/generated")
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    audit_tex.main(["--results", a.results, "--paper", str(out), "--dataset", "diabetes"])
    df = load(a.results, "diabetes")
    df["wakeups"] = [np.mean([sum(1 for b in rb if (b.get("client_bits") or [0] * 64)[k] > 0)
                              for k in range(len((rb[0].get("client_bits") or [])))]) if rb and rb[0].get("client_bits")
                     else np.nan for rb in df["_round_bits"]]
    tm = timing_table(df)
    chk = df[df.setting.str.endswith("_co") & (df.name == "fedper")]
    fed_same = None
    if len(chk):
        fed_same = True
        for r in chk.itertuples():
            ref = df[(df.setting == r.setting[:-3]) & (df.name == "fedper") & (df.seed == r.seed) & df.eps.isna()]
            cols = [PRIMARY, "valsel_round", "fed_auroc_final", "total_upload_mb", "fed_auroc_last10_mean"]
            fed_same &= bool(len(ref) == 1 and all(getattr(r, c) == ref.iloc[0][c] for c in cols))
        df = df[~(df.setting.str.endswith("_co") & (df.name == "fedper"))]
    base = df[(df.setting != "derma") & ~df.setting.str.startswith("tm_")]
    sp = base[(base.setting == "specialty") & base.eps.isna()]
    gate = gate_table(base); pk = peak_table(base); mt = main_table(base)

    class _W(TexWriter):
        def finish(self):
            (self.dir / "numbers_tmlcn.tex").write_text("".join(self.numbers) + "".join(
                f"\\newcommand{{\\flag{k}}}{{{v}}}\n" for k, v in self.flags.items()))
    W = _W(out)
    W.table("tm_main", tab_main(sp, mt, MAIN_ORDER + ["central", "hgb"]))
    pr = paired_rows(sp, MAIN_ORDER)
    W.table("tm_paired", tab_paired(pr))
    W.table("tm_gate", tab_gate(base, gate))
    W.table("tm_network", tab_network(sp, pk))
    W.table("tm_dp", tab_dp(df))
    W.table("tm_timing", tab_timing(tm) if len(tm) else None)
    fam_tex, dfam = tab_families(df)
    W.table("tm_families", fam_tex)
    hp = base[(base.setting == "hospitals") & base.eps.isna()]
    if len(hp):
        W.table("tm_hospitals", tab_main(hp, main_table(base, "hospitals", HOSP_ORDER), HOSP_ORDER, with_ft=False,
                                         worst="Worst centre"))
    sel = selection_frame(base, gate)
    W.table("tm_selection", tab_selection(sel))
    tfd = df[(df.name == "tierfed_dp")]
    pg = None
    if len(tfd):
        sch = json.loads(Path(sorted(tfd["file"])[0]).read_text()).get("tier_report") or {}
        ws, wm = sch["slow"]["params"], sch["medium"]["params"]
        pg = planner_grid(ws / (ws + wm), wm / (ws + wm))
        W.table("tm_planner", tab_planner(pg))
    # the two synchronisation steps
    sf = sync_frame(base); sg = sync_gate(base); dd = dispersion_frame(a.results)
    if len(sf):
        W.table("tm_sync", tab_sync(sf))
    if len(dd):
        W.table("tm_dispersion", tab_dispersion(dd))
    sync_numbers(sf, sg, dd, W)
    if fed_same is not None:
        W.flag("tmSyncFedperSame", fed_same)
    sync_vs_fedper(base, W); sync_resnet(base, W)
    # numbers the paired table's sentences quote
    for r in pr:
        key = "tmPair" + _safe(r["name"].title().replace("_", ""))
        W.num(key + "D", fs(r["diff"])); W.num(key + "Lo", fs(r["lo"])); W.num(key + "Hi", fs(r["hi"]))
        W.num(key + "P", fp(r["p_holm"]).replace("$", "")); W.num(key + "Higher", f"{r['higher']}/{r['k']}")
        W.num(key + "MbAll", f4(abs(r["dmb"]), 1)); W.num(key + "MbSel", f4(abs(r["dmb_sel"]), 1))
    rest = [r for r in pr if r["name"] not in ("local", "adacomm8")]
    if rest:
        W.num("tmPairRestMaxAbs", f4(max(abs(r["diff"]) for r in rest)))
        W.flag("tmPairRestNoneSig", all(r["p_holm"] >= 0.05 for r in rest))
    numbers(df, base, gate, tm, sel, pg, W, a.results, n_runs=dfam.attrs.get("total", len(df)))
    W.finish()
    print("TMLCN tables and numbers written to", out)


if __name__ == "__main__":
    main()

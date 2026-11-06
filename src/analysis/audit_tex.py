"""LaTeX tables and number macros for the v6 audit results.

  python -m src.analysis.audit_tex --results results --paper paper/generated

Writes au_main.tex, au_gate.tex, au_dp.tex, au_timing.tex, au_peak.tex and
numbers_au.tex. Everything is computed by src/analysis/audit_report.py from the
"_au_" logs; nothing is typed by hand. Selection is on validation throughout.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .audit_report import (MAIN_ORDER, PRIMARY, dp_table, gate_table, load, main_table, paired,
                           peak_table, timing_table, _uniform_k)
from .collect import TexWriter, _safe

NAMES = {"local": "Local only", "fedavg": "FedAvg", "fedprox": "FedProx", "scaffold": "SCAFFOLD (Adam)",
         "fedper": "FedPer", "fedlama": "FedLAMA-style", "adacomm8": "AdaComm", "lag_uniform": "LAG, equal weights",
         "lag": "LAG", "fixed52_legacy": "Fixed (5,2), legacy rule", "fixed52": "Fixed (5,2)",
         "fixed44": "Fixed (4,4)", "fixed44_st": "Fixed (4,4), staggered", "fixed88": "Fixed (8,8)",
         "tierfed": "\\tierfed{}", "tierfed_st": "\\tierfed{}, staggered", "tierfed_db": "\\tierfed{}, debiased",
         "tierfed_pk": "\\tierfed{}, peak cap 0.6", "central": "Centralised MLP", "hgb": "Gradient boosting",
         "tierfed_dp": "\\tierfed{}-DP", "oneshot": "One-shot", "lag_xi100": "LAG ($\\xi=100$)",
         "lag_xi2000": "LAG ($\\xi=2000$)", "lag_xi5000": "LAG ($\\xi=5000$)", "lag_xi10000": "LAG ($\\xi=10^4$)",
         "fixed22": "Fixed (2,2)", "fixed66": "Fixed (6,6)", "fixed15": "Fixed (1,5)", "adacomm4": "AdaComm, period 4",
         "adacomm16": "AdaComm, period 16", "tierfed_rho0.15": "\\tierfed{}, $\\rho=0.15$",
         "tierfed_rho0.4": "\\tierfed{}, $\\rho=0.4$", "tierfed_all": "\\tierfed{}, quota over all rounds",
         "tierfed_st_w8": "\\tierfed{}, staggered, 8-bit warm-up", "fixed44_st_w8": "Fixed (4,4), staggered, 8-bit warm-up"}
SETTING = {"specialty": "Specialty sites", "a0.1": "Dirichlet $\\alpha=0.1$", "a0.3": "Dirichlet $\\alpha=0.3$",
           "a1": "Dirichlet $\\alpha=1$", "a10": "Dirichlet $\\alpha=10$", "resnet": "ResNet backbone",
           "transformer": "FT-Transformer backbone"}


def f4(x, nd=4):
    return "--" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def fs(x, nd=4):
    """Signed difference with a typographic minus, or -- when undefined."""
    if x is None or not np.isfinite(x):
        return "--"
    v = f"{abs(x):.{nd}f}"
    return ("\\ensuremath{-}" if x < 0 and float(v) != 0 else "\\ensuremath{+}") + v


def fp(p):
    if p is None or not np.isfinite(p):
        return "--"
    return "$<10^{-3}$" if p < 1e-3 else f"{p:.3f}"


def ci_mean(v, n_boot=10000):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if len(v) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(0)
    b = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(1)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def k_label(name):
    k = _uniform_k(name)
    return "$K=1$ (FedPer)" if k == 1 else (f"$K={k}$" if k else name)


def tab_main(df, mt):
    sp = df[(df.setting == "specialty") & df.eps.isna()]
    rows = []
    for n in [x for x in MAIN_ORDER + ["central", "hgb"] if x in mt.index]:
        r = mt.loc[n]; v = sp[sp.name == n][PRIMARY]
        lo, hi = ci_mean(v)
        ece = f"{f4(r.get('fed_ece_valsel_mean'), 3)}/{f4(r.get('fed_cal_ece_valsel_mean'), 3)}"
        rows.append(" & ".join([
            NAMES.get(n, n), f"{f4(r.get(PRIMARY + '_mean'))} [{f4(lo)}, {f4(hi)}]",
            f4(r.get("total_upload_mb_mean"), 1), f4(r.get("upload_mb_valsel_mean"), 1),
            f4(r.get("fed_auroc_last10_mean_mean")), f4(r.get("fed_auroc_phase_mean")),
            f4(r.get("fed_ft3_auroc_final_mean")), ece, f4(r.get("fed_auroc_site_min_valsel_mean"), 3),
            ("--" if n == "fedper" else fs(r.get("d_fedper_valsel", np.nan))),
            ("--" if n == "fedper" else fp(r.get("p_fedper_valsel")))]) + r"\\")
    head = (r"\begin{tabular}{@{}lcccccccccc@{}}" "\n" r"\toprule" "\n"
            r"Method & AUROC, val.-selected [95\% CI] & MB & MB to sel. & Last 10 & Phase & +FT 3 & ECE raw/Platt & Worst site & $\Delta$ FedPer & $p$\\"
            "\n" r"\midrule" "\n")
    return head + "\n".join(rows) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n"


def early_k(d):
    """TierFed's intervals in the first round after warm-up, joint mode over seeds."""
    ks = []
    for f in d.get("file", []):
        try:
            h = pd.read_csv(str(f).replace("_summary.json", "_history.csv"))
        except FileNotFoundError:
            continue
        if "K_slow" in h and len(h) > 5:
            ks.append((int(h["K_slow"].iloc[5]), int(h["K_medium"].iloc[5])))
    if not ks:
        return "--"
    m = max(set(ks), key=ks.count)
    return f"({m[0]},{m[1]})"


def tab_gate(df, gate):
    rows = []
    for _, r in gate.iterrows():
        d = df[(df.setting == r.setting) & (df.name == "tierfed")]
        psi = d["psi"].mean() if "psi" in d else np.nan
        kk = early_k(d)
        rows.append(" & ".join([
            SETTING.get(r.setting, r.setting), f4(100 * psi, 1) if np.isfinite(psi) else "--", kk,
            f4(r.tierfed_auroc), f4(r.tierfed_mb, 0), k_label(r.best_uniform) if r.best_uniform else "--",
            f4(r.best_uniform_auroc), f4(r.best_uniform_mb, 0),
            f"{fs(r.gain_vs_uniform)} [{fs(r.gain_lo)}, {fs(r.gain_hi)}]" if np.isfinite(r.gain_vs_uniform) else "--",
            fp(r.get("p_vs_uniform", np.nan)), str(int(r.get("n_pairs", 0)))]) + r"\\")
    head = (r"\begin{tabular}{@{}lcccccccccc@{}}" "\n" r"\toprule" "\n"
            r"Setting & $\Psi$ (\%) & $\mathbf K$ & \tierfed{} & MB & Best uniform & AUROC & MB & Gain [95\% CI] & $p$ & Seeds\\"
            "\n" r"\midrule" "\n")
    return head + "\n".join(rows) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n"


def tab_dp(df, dp):
    """q = 1, epsilon* = 4: releases, calibrated noise multiplier, clipped share and AUROC."""
    if dp is None or dp.empty:
        return None
    d = dp[(dp.q == 1.0) & (dp.eps == 4.0)].set_index("name")
    order = ["fedper", "fixed44", "fixed88", "oneshot", "tierfed_dp"]
    rows = []
    for n in [o for o in order if o in d.index]:
        r = d.loc[n]
        name = NAMES.get(n, n.replace("fixed44", "Fixed (4,4)").replace("fixed88", "Fixed (8,8)"))
        rows.append(" & ".join([name, f4(r.get("dp_rounds_charged_mean"), 0), f4(r.get("dp_sigma_mean"), 2),
                                f4(r.get("clip_fraction_mean"), 2), f4(r.get(PRIMARY + "_mean")),
                                f4(r.get("valsel_round_mean"), 0), f4(r.get("fed_auroc_final_mean"))]) + r"\\")
    la = dp.attrs.get("local_auroc", (np.nan, np.nan))
    rows.append(r"\midrule" "\n" f"Local only ($\\varepsilon=0$) & 0 & -- & -- & {f4(la[0])} & -- & --" r"\\")
    head = (r"\begin{tabular}{@{}lcccccc@{}}" "\n" r"\toprule" "\n"
            r"Method & Releases & $z$ & Clipped & AUROC & Round & Final\\" "\n" r"\midrule" "\n")
    return head + "\n".join(rows) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n"


def tab_timing(tm):
    if tm is None or tm.empty:
        return None
    rows = [" & ".join([NAMES.get(r["name"], r["name"]), f4(r.auc_tx_rounds, 2), f4(r.auc_target_tx_rounds, 2),
                        f4(r.auc_target_mb, 2), f4(r.auc_total_mb, 2),
                        str(int(r.n_clean)) + "/" + str(int(r.n_shift))]) + r"\\"
            for _, r in tm.iterrows()]
    head = (r"\begin{tabular}{@{}lccccc@{}}" "\n" r"\toprule" "\n"
            r"Schedule & Traffic rounds & Target rounds & Target bytes & All bytes & Runs\\"
            "\n" r"\midrule" "\n")
    return head + "\n".join(rows) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n"


def tab_peak(pk, df):
    if pk is None or pk.empty:
        return None
    sp = df[(df.setting == "specialty") & df.eps.isna()]
    rows = []
    for n, r in pk.iterrows():
        wake = sp[sp.name == n]["wakeups"].mean() if "wakeups" in sp else np.nan
        rows.append(" & ".join([NAMES.get(n, n), f4(r.peak_frac, 2), f4(r["violation_rate_W0.6"], 2),
                                f4(r["violation_rate_W0.8"], 2), f4(wake, 0)]) + r"\\")
    head = (r"\begin{tabular}{@{}lcccc@{}}" "\n" r"\toprule" "\n"
            r"Method & Peak/payload & Viol., $W=0.6t_1$ & Viol., $W=0.8t_1$ & Wake-ups/site\\" "\n" r"\midrule" "\n")
    return head + "\n".join(rows) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n"


def numbers(df, mt, gate, dp, tm, W):
    base = df
    for n in [x for x in MAIN_ORDER + ["central", "hgb"] if x in mt.index]:
        key = "au" + _safe(n.title().replace("_", ""))
        r = mt.loc[n]
        W.num(key + "Auroc", f4(r.get(PRIMARY + "_mean")))
        W.num(key + "Mb", f4(r.get("total_upload_mb_mean"), 1))
        W.num(key + "MbSel", f4(r.get("upload_mb_valsel_mean"), 1))
        W.num(key + "Last", f4(r.get("fed_auroc_last10_mean_mean")))
        W.num(key + "Ftthree", f4(r.get("fed_ft3_auroc_final_mean")))
        W.num(key + "EceRaw", f4(r.get("fed_ece_valsel_mean"), 3))
        W.num(key + "EceCal", f4(r.get("fed_cal_ece_valsel_mean"), 3))
        W.num(key + "Final", f4(r.get("fed_auroc_final_mean")))
        W.num(key + "Auprc", f4(r.get("fed_auprc_valsel_mean"), 3))
        W.num(key + "Phase", f4(r.get("fed_auroc_phase_mean")))
        W.num(key + "Round", f4(r.get("valsel_round_mean"), 0))
        W.num(key + "Worst", f4(r.get("fed_auroc_site_min_valsel_mean"), 3))
        W.num(key + "Slope", f4(r.get("fed_cal_slope_valsel_mean"), 2))
        if n != "fedper":
            W.num(key + "DFedper", fs(r.get("d_fedper_valsel", np.nan)))
            W.num(key + "DFedperP", fp(r.get("p_fedper_valsel")).replace("$", ""))
    fed = [n for n in MAIN_ORDER if n in mt.index and n not in ("local", "fedper", "adacomm8")]
    if fed:
        d = mt.loc[fed, "d_fedper_valsel"].abs()
        W.num("auFedSpread", f4(d.max(), 4)); W.num("auFedSpreadName", NAMES.get(d.idxmax(), d.idxmax()))
        W.num("auFedMinP", fp(mt.loc[fed, "p_fedper_valsel"].min()).replace("$", ""))
    present = [n for n in MAIN_ORDER if n in mt.index]
    w = mt.loc[present, "fed_auroc_site_min_valsel_mean"]
    W.num("auWorstMin", f4(w.min(), 3)); W.num("auWorstMax", f4(w.max(), 3))
    e = mt.loc[present, "fed_cal_ece_valsel_mean"]; er = mt.loc[present, "fed_ece_valsel_mean"]
    W.num("auEceCalMin", f4(e.min(), 3)); W.num("auEceCalMax", f4(e.max(), 3))
    W.num("auEceRawMin", f4(er.min(), 3)); W.num("auEceRawMax", f4(er.max(), 3))
    ft = mt.loc[present, "fed_ft3_auroc_final_mean"] - mt.loc[present, "fed_auroc_final_mean"]
    W.num("auFtLossMin", f4(-ft.max(), 4)); W.num("auFtLossMax", f4(-ft.min(), 4))
    W.flag("auFtAlwaysLowers", bool((ft < 0).all()))
    sp = df[(df.setting == "specialty") & df.eps.isna()]
    ts = sp[sp.name == "tierfed"]
    if len(ts) and "sch_K_slow" in ts:
        W.flag("auTierfedAlwaysFourFour", bool(((ts["sch_K_slow"] == 4) & (ts["sch_K_medium"] == 4)).all()))
        a = sp[sp.name == "tierfed"].set_index("seed")[PRIMARY]; b = sp[sp.name == "fixed44"].set_index("seed")[PRIMARY]
        common = a.index.intersection(b.index)
        W.flag("auTierfedEqualsFixed", bool(len(common) and (a[common] == b[common]).all()))
    for n in ("tierfed", "fixed44", "adacomm8", "fixed88"):
        if n in mt.index:
            key = "au" + _safe(n.title().replace("_", ""))
            W.num(key + "NiTenP", fp(mt.loc[n].get("p_ni010_fedper_valsel", np.nan)).replace("$", ""))
            W.num(key + "NiFiveP", fp(mt.loc[n].get("p_ni005_fedper_valsel", np.nan)).replace("$", ""))
            W.num(key + "Seeds", str(int(mt.loc[n].get("n_seeds", 0))))
    for _, r in gate.iterrows():
        key = "auGate" + _safe(r.setting.replace(".", "p").title())
        W.num(key + "Gain", fs(r.gain_vs_uniform))
        W.num(key + "P", fp(r.get("p_vs_uniform", np.nan)).replace("$", ""))
        W.num(key + "PHolm", fp(r.get("p_vs_uniform_holm", np.nan)).replace("$", ""))
        W.num(key + "Lo", fs(r.gain_lo)); W.num(key + "Hi", fs(r.gain_hi))
        W.flag(key + "Wins", bool(np.isfinite(r.gain_lo) and r.gain_lo > 0 and r.gain_vs_uniform >= 0.005))
        W.flag(key + "Loses", bool(np.isfinite(r.gain_hi) and r.gain_hi < 0 and r.gain_vs_uniform <= -0.005))
        W.num(key + "TfAuroc", f4(r.tierfed_auroc)); W.num(key + "UAuroc", f4(r.best_uniform_auroc))
        W.num(key + "TfMb", f4(r.tierfed_mb, 0)); W.num(key + "UMb", f4(r.best_uniform_mb, 0))
        W.num(key + "FedperAuroc", f4(r.fedper_auroc)); W.num(key + "LocalAuroc", f4(r.local_auroc))
        W.num(key + "Seeds", str(int(r.get("n_pairs", 0))))
        W.num(key + "Best", k_label(r.best_uniform) if r.best_uniform else "--")
        d = df[(df.setting == r.setting) & (df.name == "tierfed")]
        if "psi" in d and d.psi.notna().any():
            W.num(key + "Psi", f4(100 * d.psi.mean(), 1))
        if "sch_frac_rounds_tier_differentiated" in d:
            W.num(key + "Diff", f4(d["sch_frac_rounds_tier_differentiated"].mean(), 2))
    het = gate[gate.setting.str.startswith("a")]
    if len(het):
        psis, rates = [], []
        for st in het.setting:
            d = df[(df.setting == st) & (df.name == "tierfed")]
            if "psi" in d and d.psi.notna().any():
                psis.append(d.psi.mean())
            if "sch_frac_rounds_tier_differentiated" in d:
                rates.append(d["sch_frac_rounds_tier_differentiated"].mean())
        if psis:
            W.num("auHetPsiMax", f4(100 * max(psis), 1))
            W.flag("auHetPsiSmall", bool(max(psis) < 0.05))
        if rates:
            W.num("auHetDiffMax", f4(max(rates), 2))
        W.num("auHetGainMin", fs(het.gain_vs_uniform.min())); W.num("auHetGainMax", fs(het.gain_vs_uniform.max()))
        W.num("auHetSettings", str(len(het)))
        W.flag("auHetAnyWin", bool(((het.gain_lo > 0) & (het.gain_vs_uniform >= 0.005)).any()))
        W.flag("auHetAnyLose", bool(((het.gain_hi < 0) & (het.gain_vs_uniform <= -0.005)).any()))
        # sites that receive data, and test splits with one class (excluded for every method alike)
        st0 = het.setting.iloc[0]          # the strongest skew (settings are sorted by alpha)
        d0 = df[(df.setting == st0) & df.eps.isna()]
        nsites = [len(rb[0].get("client_bits") or []) for rb in d0["_round_bits"] if rb]
        if nsites:
            W.num("auHetSkewAlpha", st0[1:])
            W.num("auHetSitesMin", str(min(nsites))); W.num("auHetSitesMax", str(max(nsites)))
            W.flag("auHetSitesAll", min(nsites) == 20)
        if "n_clients_single_class_valsel" in d0 and d0["n_clients_single_class_valsel"].notna().any():
            sc = d0["n_clients_single_class_valsel"].dropna()
            W.num("auHetSingleMin", str(int(sc.min()))); W.num("auHetSingleMax", str(int(sc.max())))
    rn = df[(df.setting == "resnet") & (df.name == "tierfed")]
    if len(rn) and "sch_K_slow" in rn:
        k = rn[["sch_K_slow", "sch_K_medium"]].dropna().astype(int)
        if len(k):
            mode = k.value_counts().index[0]
            W.num("auResnetK", f"({mode[0]},{mode[1]})")
        if "sch_frac_rounds_tier_differentiated" in rn:
            W.num("auResnetDiff", f4(rn["sch_frac_rounds_tier_differentiated"].mean(), 2))
        # intervals right after warm-up, the cube-root prediction, and where the adaptive
        # schedule first departs from the frozen (1,5) control
        early, first_diff, sel_ok, ratios = [], [], [], []
        for _, row in rn.iterrows():
            h = pd.read_csv(row["file"].replace("_summary.json", "_history.csv"))
            if "K_slow" in h and len(h) > 5:
                early.append((int(h["K_slow"].iloc[5]), int(h["K_medium"].iloc[5])))
            from .theory_numbers import warmup_psi
            wp = warmup_psi(row["file"])
            if wp:
                ratios.append(wp["ratio"])
            fz = row["file"].replace("_tierfed_R60", "_fixed_R60").replace("_au_bb_tierfed_", "_au_bb_fixed15_")
            try:
                hf = pd.read_csv(fz.replace("_summary.json", "_history.csv"))
                d = (h["fed_auroc"] - hf["fed_auroc"]).abs() > 0
                fd = int(d.idxmax()) + 1 if d.any() else len(h) + 1
                first_diff.append(fd); sel_ok.append(fd > row["valsel_round"])
            except FileNotFoundError:
                pass
        if early:
            mode = max(set(early), key=early.count)
            W.num("auResnetKEarly", f"({mode[0]},{mode[1]})")
            W.flag("auResnetKEarlyAll", all(e == mode for e in early))
        if ratios:
            W.num("auResnetRatio", f4(float(np.mean(ratios)), 1))
        if first_diff:
            W.num("auResnetFirstDiffMin", str(min(first_diff)))
            W.flag("auResnetFrozenSameAtSel", bool(all(sel_ok)))
        rd = df[df.setting == "resnet"]
        for other, lab in (("fixed15", "Frozen"), ("fedper", "Fedper"), ("local", "Local")):
            if other in set(rd.name):
                r = paired(rd, "tierfed", other)
                W.num("auResnetVs" + lab, fs(r["diff"])); W.num("auResnetVs" + lab + "Lo", fs(r["lo"]))
                W.num("auResnetVs" + lab + "Hi", fs(r["hi"])); W.num("auResnetVs" + lab + "P", fp(r["p"]).replace("$", ""))
        for n in ("tierfed", "fixed44", "fixed15", "fedper", "local"):
            v = rd[rd.name == n]
            if len(v):
                key = "auResnet" + _safe(n.title())
                W.num(key + "Auroc", f4(v[PRIMARY].mean())); W.num(key + "Mb", f4(v["total_upload_mb"].mean(), 0))
    won = gate[(gate.gain_lo > 0) & (gate.gain_vs_uniform >= 0.005)]   # interval above zero, relevant size
    W.num("auSettingsWon", str(len(won))); W.num("auSettings", str(len(gate)))
    W.flag("auRouteA", len(won) > 0)
    if tm is not None and not tm.empty:
        for _, r in tm.iterrows():
            key = "auTiming" + _safe(r["name"].title().replace("_", ""))
            W.num(key + "AucTraffic", f4(r.auc_tx_rounds, 2))
            W.num(key + "AucTarget", f4(r.auc_target_tx_rounds, 2))
            W.num(key + "AucBytes", f4(r.auc_target_mb, 2))
            W.num(key + "Seeds", str(int(min(r.n_clean, r.n_shift))))
            aucs = [r.auc_tx_rounds, r.auc_target_tx_rounds, r.auc_target_mb, r.auc_total_mb]
            W.num(key + "AucAll", f4(r.auc_total_mb, 2))
            W.num(key + "OtherSkipPct", f4(100 * r.other_skip_share, 0))
            W.flag(key + "TargetSkips", bool(r.target_skips > 0))
            W.flag(key + "TargetLeaks", bool(np.nanmax(aucs[:3]) >= 0.75))
            W.flag(key + "Leaks", bool(np.nanmax(aucs) >= 0.75))
            W.flag(key + "Silent", bool(all(abs(a - 0.5) < 1e-9 for a in aucs)))
            W.num(key + "AucMax", f4(float(np.nanmax(aucs)), 2))
        lag = tm[tm["name"].str.startswith("lag")]
        if len(lag):
            best = float(np.nanmax(lag[["auc_tx_rounds", "auc_target_tx_rounds", "auc_target_mb",
                                        "auc_total_mb"]].to_numpy()))
            W.flag("auTimingLagLeaks", best >= 0.75)
    if dp is not None and not dp.empty:
        la = dp.attrs.get("local_auroc", (np.nan, np.nan))
        W.num("auDpLocal", f4(la[0]))
        d1 = dp[(dp.q == 1.0) & (dp.eps == 4.0)].set_index("name")
        for n in d1.index:
            key = "auDp" + _safe(n.title().replace("_", ""))
            W.num(key + "Auroc", f4(d1.loc[n, PRIMARY + "_mean"]))
            if "dp_rounds_charged_mean" in d1:
                W.num(key + "Releases", f4(d1.loc[n, "dp_rounds_charged_mean"], 0))
            if "dp_sigma_mean" in d1:
                W.num(key + "Z", f4(d1.loc[n, "dp_sigma_mean"], 2))
            if "clip_fraction_mean" in d1:
                W.num(key + "Clip", f4(d1.loc[n, "clip_fraction_mean"], 2))
            W.num(key + "Round", f4(d1.loc[n, "valsel_round_mean"], 0))
            W.num(key + "Final", f4(d1.loc[n, "fed_auroc_final_mean"]))
        one = base[(base.name == "oneshot") & (base.eps == 4.0) & (base.q == 1.0)]
        if len(one):
            W.flag("auDpOneshotPreRelease", bool((one["valsel_round"] < one["rounds"]).all()))
        # every arm that releases before its selected round: the best of them
        rel = d1[d1["valsel_round_mean"] > 1]
        W.num("auDpBestReleasing", f4(d1.drop(index=[i for i in d1.index if i == "oneshot"])[PRIMARY + "_mean"].max()))
        d = dp[dp.q == 1.0]
        for e in sorted(d.eps.unique()):
            x = d[d.eps == e].set_index("name")[PRIMARY + "_mean"]
            if len(x):
                W.num(f"auDpBestEps{_safe(str(int(e)))}", f4(x.max()))
                W.num(f"auDpBestEps{_safe(str(int(e)))}Name", NAMES.get(x.idxmax(), x.idxmax()))


def frontier_numbers(base, W, floor_mb=40.0):
    """Plateau of the accuracy-upload plane on the seeds every configuration shares."""
    from .audit_report import common_seeds
    sp = base[(base.setting == "specialty") & base.eps.isna() & ~base.name.isin(["central", "hgb"])]
    seeds = common_seeds(sp)
    sp = sp[sp.seed.isin(seeds)]
    if sp.empty:
        return
    m = sp.groupby("name")[[PRIMARY, "total_upload_mb"]].mean()
    plat = m[m.total_upload_mb >= floor_mb]
    W.num("auFrontierSeeds", str(len(seeds)))
    W.num("auPlateauFrom", f"{floor_mb:g}")
    W.num("auPlateauMin", f4(plat[PRIMARY].min())); W.num("auPlateauMax", f4(plat[PRIMARY].max()))
    W.num("auPlateauN", str(len(plat)))
    for a, b, lab in (("adacomm4", "tierfed", "AdaFourVsTf"), ("tierfed_rho0.15", "fixed88", "TfFifteenVsEight")):
        if a in m.index and b in m.index:
            r = paired(sp, a, b)
            W.num("au" + lab, fs(r["diff"])); W.num("au" + lab + "Lo", fs(r["lo"])); W.num("au" + lab + "Hi", fs(r["hi"]))
            W.num("au" + lab + "P", fp(r["p"]).replace("$", ""))
            W.num("au" + lab + "MbA", f4(m.loc[a, "total_upload_mb"], 1)); W.num("au" + lab + "MbB", f4(m.loc[b, "total_upload_mb"], 1))


def more_numbers(budget, pk, base, W):
    if budget is not None and not budget.empty:
        for _, r in budget.iterrows():
            if r.get("config") is None or (isinstance(r.get("config"), float) and np.isnan(r.get("config"))):
                continue
            key = "auBudget" + _safe(r["family"].title().replace(" ", "").replace("-", "")) + _safe(str(int(r["budget_mb"])))
            W.num(key, f4(r["auroc_mean"]))
            W.num(key + "Config", NAMES.get(r["config"], str(r["config"]).replace("_", " ")))
    if pk is not None and not pk.empty:
        for n, r in pk.iterrows():
            key = "auPeak" + _safe(n.title().replace("_", ""))
            W.num(key + "Frac", f4(r.peak_frac, 2))
            W.num(key + "Post", f4(r.get("peak_post_frac", np.nan), 2))
            W.num(key + "ViolEight", f4(r["violation_rate_W0.8"], 2))
            W.num(key + "ViolSix", f4(r["violation_rate_W0.6"], 2))
        sp = base[(base.setting == "specialty") & base.eps.isna()]
        for n in ("tierfed", "tierfed_st", "fixed44", "fixed44_st", "fedper", "tierfed_st_w8", "fixed44_st_w8"):
            v = sp[sp.name == n]
            if len(v):
                key = "auPeak" + _safe(n.title().replace("_", ""))
                W.num(key + "Wake", f4(v["wakeups"].mean(), 0))
                W.num(key + "PeakAll", f4(v["peak_frac"].mean(), 2))
        for n in ("fedper", "tierfed", "adacomm8", "fixed88"):
            v = sp[sp.name == n]
            key = "auNet" + _safe(n.title().replace("_", ""))
            for col, lab in (("net_nr_cell_edge_comm_time_s", "Time"), ("net_nr_cell_edge_shared_comm_time_s", "Shared"),
                             ("net_nr_cell_edge_energy_j", "Energy")):
                if col in v and v[col].notna().any():
                    W.num(key + lab, f4(v[col].mean(), 0 if lab != "Energy" else 1))
        d = paired(sp, "fixed44_st", "fixed44")
        W.num("auStaggerDiff", fs(d["diff"])); W.num("auStaggerP", fp(d["p"]).replace("$", ""))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--paper", default="paper/generated")
    ap.add_argument("--dataset", default="diabetes")
    a = ap.parse_args(argv)
    out = Path(a.paper); out.mkdir(parents=True, exist_ok=True)
    df = load(a.results, a.dataset)
    if df.empty:
        print("no _au_ logs"); return
    n_runs = len(df)          # every v6 run, all settings
    df["wakeups"] = [np.mean([sum(1 for b in rb if (b.get("client_bits") or [0] * 64)[k] > 0)
                              for k in range(len((rb[0].get("client_bits") or [])))]) if rb and rb[0].get("client_bits")
                     else np.nan for rb in df["_round_bits"]]
    tm = timing_table(df)
    base = df[(df.setting != "derma") & ~df.setting.str.startswith("tm_")]
    mt, gate, pk, dp = main_table(base), gate_table(base), peak_table(base), dp_table(base)

    class _W(TexWriter):
        def finish(self):
            (self.dir / "numbers_au.tex").write_text("".join(self.numbers) + "".join(
                f"\\newcommand{{\\flag{k}}}{{{v}}}\n" for k, v in self.flags.items()))
    W = _W(out)
    W.table("au_main", tab_main(base, mt))
    W.table("au_gate", tab_gate(base, gate))
    W.table("au_dp", tab_dp(base, dp))
    W.table("au_timing", tab_timing(tm))
    W.table("au_peak", tab_peak(pk, base))
    numbers(base, mt, gate, dp, tm, W)
    from .audit_report import frontier_tables
    _, budget = frontier_tables(base)
    more_numbers(budget, pk, base, W)
    frontier_numbers(base, W)
    W.num("auRunsVsix", f"{n_runs:,}")
    # cohort counts, read from a v6 log (every v6 run uses the same cleaned cohort)
    f0 = next(iter(sorted(base[base.setting == "specialty"]["file"])), None)
    if f0:
        ds = json.loads(Path(f0).read_text()).get("dataset", {})
        if ds.get("n_encounters_after_cleaning"):
            W.num("auNEncounters", f"{int(ds['n_encounters_after_cleaning']):,}")
        W.num("auNSitePool", f"{int(ds['n_train']):,}"); W.num("auNHoldout", f"{int(ds['n_test']):,}")
    W.finish()
    print("audit tables written to", out)


if __name__ == "__main__":
    main()

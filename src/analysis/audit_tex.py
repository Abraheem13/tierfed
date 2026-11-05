"""LaTeX tables and number macros for the v6 audit results.

  python -m src.analysis.audit_tex --results results --paper paper/generated

Writes au_main.tex, au_gate.tex, au_dp.tex, au_timing.tex, au_peak.tex and
numbers_au.tex. Everything is computed by src/analysis/audit_report.py from the
"_au_" logs; nothing is typed by hand. Selection is on validation throughout.
"""
from __future__ import annotations

import argparse
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
         "tierfed_dp": "\\tierfed{}-DP", "oneshot": "One-shot", "lag_xi100": "LAG ($\\xi=100$)"}
SETTING = {"specialty": "Specialty sites", "a0.1": "Dirichlet $\\alpha=0.1$", "a0.3": "Dirichlet $\\alpha=0.3$",
           "a1": "Dirichlet $\\alpha=1$", "a10": "Dirichlet $\\alpha=10$", "resnet": "ResNet backbone",
           "transformer": "FT-Transformer backbone"}


def f4(x, nd=4):
    return "--" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def fs(x, nd=4):
    """Signed difference, or -- when undefined."""
    return "--" if x is None or not np.isfinite(x) else f"{x:+.{nd}f}"


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


def tab_gate(df, gate):
    rows = []
    for _, r in gate.iterrows():
        d = df[(df.setting == r.setting) & (df.name == "tierfed")]
        psi = d["psi"].mean() if "psi" in d else np.nan
        ks = d.get("sch_K_slow"); km = d.get("sch_K_medium")
        kk = (f"({int(ks.mode().iloc[0])},{int(km.mode().iloc[0])})"
              if ks is not None and ks.notna().any() and km.notna().any() else "--")
        rows.append(" & ".join([
            SETTING.get(r.setting, r.setting), f4(100 * psi, 1) if np.isfinite(psi) else "--", kk,
            f4(r.tierfed_auroc), f4(r.tierfed_mb, 0), k_label(r.best_uniform) if r.best_uniform else "--",
            f4(r.best_uniform_auroc), f4(r.best_uniform_mb, 0),
            f"{fs(r.gain_vs_uniform)} [{fs(r.gain_lo)}, {fs(r.gain_hi)}]" if np.isfinite(r.gain_vs_uniform) else "--",
            fp(r.get("p_vs_uniform_holm", np.nan))]) + r"\\")
    head = (r"\begin{tabular}{@{}lccccccccc@{}}" "\n" r"\toprule" "\n"
            r"Setting & $\Psi$ (\%) & $\mathbf K$ & \tierfed{} & MB & Best uniform & AUROC & MB & Gain [95\% CI] & $p_{\mathrm{Holm}}$\\"
            "\n" r"\midrule" "\n")
    return head + "\n".join(rows) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n"


def tab_dp(df, dp):
    if dp is None or dp.empty:
        return None
    d = dp[dp.q == 1.0]
    order = ["fedavg", "fedper", "fixed44", "fixed88", "fixed1212", "fixed2020", "fixed3030", "oneshot", "tierfed_dp"]
    eps = sorted(d.eps.unique())
    rows = []
    for n in [o for o in order if o in set(d.name)]:
        x = d[d.name == n].set_index("eps")
        rel = x["dp_rounds_charged_mean"].iloc[0] if "dp_rounds_charged_mean" in x else np.nan
        cells = [f"{f4(x.loc[e, PRIMARY + '_mean'])}" if e in x.index else "--" for e in eps]
        name = NAMES.get(n, n.replace("fixed", "Fixed ").replace("1212", "(12,12)").replace("2020", "(20,20)")
                         .replace("3030", "(30,30)").replace("44", "(4,4)").replace("88", "(8,8)"))
        rows.append(" & ".join([name, f4(rel, 0) if n != "tierfed_dp" else "planned"] + cells) + r"\\")
    la = dp.attrs.get("local_auroc", (np.nan, np.nan))
    rows.append(r"\midrule" "\n" f"Local only ($\\varepsilon=0$) & 0 & \\multicolumn{{{len(eps)}}}{{c}}{{{f4(la[0])}}}" r"\\")
    head = (r"\begin{tabular}{@{}lc" + "c" * len(eps) + r"@{}}" "\n" r"\toprule" "\n"
            r"Method & Releases & " + " & ".join(f"$\\varepsilon^\\star={e:g}$" for e in eps) + r"\\"
            "\n" r"\midrule" "\n")
    return head + "\n".join(rows) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n"


def tab_timing(tm):
    if tm is None or tm.empty:
        return None
    rows = [" & ".join([NAMES.get(r["name"], r["name"]), f4(r.auc_tx_rounds, 2), f4(r.auc_target_tx_rounds, 2),
                        f4(r.auc_target_mb, 2), str(int(r.n_clean)) + "/" + str(int(r.n_shift))]) + r"\\"
            for _, r in tm.iterrows()]
    head = (r"\begin{tabular}{@{}lcccc@{}}" "\n" r"\toprule" "\n"
            r"Schedule & Rounds with traffic & Target's rounds & Target's bytes & Runs\\" "\n" r"\midrule" "\n")
    return head + "\n".join(rows) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n"


def tab_peak(pk, df):
    if pk is None or pk.empty:
        return None
    sp = df[(df.setting == "specialty") & df.eps.isna()]
    rows = []
    for n, r in pk.iterrows():
        wake = sp[sp.name == n]["wakeups"].mean() if "wakeups" in sp else np.nan
        rows.append(" & ".join([NAMES.get(n, n), f4(r.peak_frac, 2), f4(r["violation_rate_W0.6"], 2),
                                f4(r["violation_rate_W0.8"], 2), f4(wake, 1)]) + r"\\")
    head = (r"\begin{tabular}{@{}lcccc@{}}" "\n" r"\toprule" "\n"
            r"Method & Peak/payload & Viol., $W=0.6t_1$ & Viol., $W=0.8t_1$ & Wake-ups/site\\" "\n" r"\midrule" "\n")
    return head + "\n".join(rows) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n"


def numbers(df, mt, gate, dp, tm, W):
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
        if n != "fedper":
            W.num(key + "DFedper", fs(r.get("d_fedper_valsel", np.nan)))
            W.num(key + "DFedperP", fp(r.get("p_fedper_valsel")).replace("$", ""))
    for _, r in gate.iterrows():
        key = "auGate" + _safe(r.setting.replace(".", "p").title())
        W.num(key + "Gain", fs(r.gain_vs_uniform))
        W.num(key + "P", fp(r.get("p_vs_uniform_holm", np.nan)).replace("$", ""))
        W.num(key + "Best", k_label(r.best_uniform) if r.best_uniform else "--")
        d = df[(df.setting == r.setting) & (df.name == "tierfed")]
        if "psi" in d and d.psi.notna().any():
            W.num(key + "Psi", f4(100 * d.psi.mean(), 1))
        if "sch_frac_rounds_tier_differentiated" in d:
            W.num(key + "Diff", f4(d["sch_frac_rounds_tier_differentiated"].mean(), 2))
    won = gate[(gate.get("p_vs_uniform_holm", pd.Series(dtype=float)) < 0.05) & (gate.gain_vs_uniform >= 0.005)]
    W.num("auSettingsWon", str(len(won))); W.num("auSettings", str(len(gate)))
    W.flag("auRouteA", len(won) > 0)
    if tm is not None and not tm.empty:
        for _, r in tm.iterrows():
            key = "auTiming" + _safe(r["name"].title().replace("_", ""))
            W.num(key + "Auc", f4(max(r.auc_tx_rounds, r.auc_target_tx_rounds, r.auc_target_mb), 2))
    if dp is not None and not dp.empty:
        la = dp.attrs.get("local_auroc", (np.nan, np.nan))
        W.num("auDpLocal", f4(la[0]))
        d = dp[dp.q == 1.0]
        for e in sorted(d.eps.unique()):
            x = d[d.eps == e].set_index("name")[PRIMARY + "_mean"]
            if len(x):
                W.num(f"auDpBestEps{_safe(str(int(e)))}", f4(x.max()))
                W.num(f"auDpBestEps{_safe(str(int(e)))}Name", NAMES.get(x.idxmax(), x.idxmax()))


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
    W.finish()
    print("audit tables written to", out)


if __name__ == "__main__":
    main()

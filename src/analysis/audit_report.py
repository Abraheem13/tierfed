"""v6 audit report: the decision gate and every audit table, from the "_au_" logs only.

  python -m src.analysis.audit_report --results results

Writes results/tables/audit/{main,frontier,budget,gate,peak,dp,derma,bounds}.csv,
gate.json and REPORT.md. Nothing here reads test data to choose anything: the
round of each run, the configuration of each method at a byte budget and the
uniform period a contrast is made against are all chosen on validation AUROC.

Primary contrasts (pre-registered, S-4), on the validation-selected test AUROC,
paired by seed: TierFed (rho = 0.25) against FedPer, and TierFed against the best
uniform period at matched bytes. Every other comparison is exploratory.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

PRIMARY = "fed_auroc_valsel"
MATCH_TOL = 1.05                 # "matched bytes": at most 5 % above TierFed's mean upload
PRACTICAL = 0.005                # smallest AUROC gain called practically relevant
BUDGETS = (30.0, 60.0, 120.0)    # MB, total over all sites

MAIN_ORDER = ["local", "fedavg", "fedprox", "fedper", "fedlama", "adacomm8", "lag_uniform", "lag",
              "fixed44", "fixed44_st", "fixed88", "tierfed", "tierfed_st"]
UNIFORM_RE = re.compile(r"^fixed(\d+)(\d+)$")


# --------------------------------------------------------------------------- #
def _uniform_k(name: str):
    """K of a uniform aligned period (fedper is K = 1), else None."""
    if name == "fedper":
        return 1
    m = re.fullmatch(r"fixed(\d)(\d)", name) or re.fullmatch(r"fixed(\d\d)(\d\d)", name)
    if m and m.group(1) == m.group(2):
        return int(m.group(1))
    return None


def family(name: str) -> str:
    if name in ("tierfed",) or name.startswith("tierfed_rho"):
        return "TierFed"
    if name.startswith("tierfed_"):
        return "TierFed variant"
    if _uniform_k(name) is not None:
        return "Uniform period"
    if re.fullmatch(r"fixed\d\d", name):
        return "Per-tier fixed"
    if name.startswith("adacomm"):
        return "AdaComm"
    if name == "lag" or name.startswith("lag_xi"):
        return "LAG"
    if name.startswith("fedlama"):
        return "FedLAMA"
    if name.startswith("fedprox"):
        return "FedProx"
    if "_q8" in name:
        return "8-bit"
    return "other"


def parse_cfg(cfg: str, strategy: str):
    """Split a tag into (setting, name, eps, q). setting: 'specialty', 'a<alpha>', 'derma'."""
    eps = q = None
    setting = "specialty"
    m = re.match(r"^het_a([0-9.]+)_(.+)$", cfg)
    if m:
        setting, cfg = f"a{float(m.group(1)):g}", m.group(2)
    m = re.match(r"^drop([0-9.]+)_(.+)$", cfg)
    if m:
        setting, cfg = f"drop{m.group(1)}", m.group(2)
    m = re.match(r"^tm_(clean|shift)_(.+)$", cfg)
    if m:
        setting, cfg = f"tm_{m.group(1)}", m.group(2)
    m = re.match(r"^bb_(.+)$", cfg)
    if m:
        setting, cfg = "backbone", m.group(1)
    m = re.match(r"^derma_a([0-9.]+)_(.+)$", cfg)
    if m:
        setting, cfg = "derma", m.group(2)
    m = re.match(r"^dpe([0-9.]+)_q([0-9.]+)_(.+)$", cfg)
    if m:
        eps, q, cfg = float(m.group(1)), float(m.group(2)), m.group(3)
    cfg = cfg.replace("_comp_quant8", "")
    if cfg == "bound":
        cfg = strategy                         # 'central' or 'hgb'
    return setting, cfg, eps, q


def _peak_stats(d, n_clients):
    """Per-site upload of transmitting rounds, as a fraction of the full shared payload."""
    rb = d.get("round_bits") or []
    tr = d.get("tier_report") or {}
    shared = sum(v["params"] for g, v in tr.items() if g in ("slow", "medium")) * 32.0
    peaks = [b["max_client_bits"] for b in rb if b.get("upload_bits", 0) > 0]
    if not peaks or not shared:
        return {"peak_frac": np.nan, "p95_frac": np.nan, "tx_bits_site": [], "shared_bits": shared}
    s = sorted(peaks)
    return {"peak_frac": s[-1] / shared, "p95_frac": s[min(len(s) - 1, int(0.95 * len(s)))] / shared,
            "tx_bits_site": [b["max_client_bits"] for b in rb], "shared_bits": shared}


def load(results: str, dataset: str | None) -> pd.DataFrame:
    rows = []
    for f in glob.glob(str(Path(results) / "logs" / "*_au*_summary.json")):
        d = json.loads(Path(f).read_text())
        c = d.get("config", {})
        if dataset and c.get("dataset") not in (dataset, "medmnist"):
            continue
        tag = c.get("tag", "")
        if not tag.startswith("_au"):
            continue
        setting, name, eps, q = parse_cfg(tag[len("_au_"):] if tag.startswith("_au_") else tag[3:].lstrip("_"),
                                          c.get("strategy", ""))
        s = d["summary"]
        if setting == "backbone":
            setting = c.get("model", "backbone")
        row = {"file": f, "setting": setting, "name": name, "eps": eps, "q": q, "seed": int(c.get("seed", -1)),
               "strategy": c.get("strategy"), "dataset": c.get("dataset"), **s}
        sch = d.get("schedule") or {}
        for k in ("frac_rounds_tier_differentiated", "K_slow", "K_medium", "K_slow_mean", "K_medium_mean"):
            if k in sch:
                row[f"sch_{k}"] = sch[k]
        for k, v in (d.get("dp") or {}).items():
            row[k] = v
        row.update({k: v for k, v in _peak_stats(d, 20).items() if k in ("peak_frac", "p95_frac", "shared_bits")})
        row["_round_bits"] = d.get("round_bits") or []
        if c.get("strategy") == "tierfed":
            from .theory_numbers import warmup_psi
            wp = warmup_psi(f)
            row["psi"] = wp["psi"] if wp else np.nan
        if eps is not None:
            h = Path(f.replace("_summary.json", "_history.csv"))
            if h.exists():
                hd = pd.read_csv(h)
                if "dp_clip_fraction" in hd:
                    rel = hd["upload_mb"].diff().fillna(hd["upload_mb"]) > 0
                    row["clip_fraction"] = float(hd.loc[rel, "dp_clip_fraction"].mean()) if rel.any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def paired(df, a_name, b_name, metric=PRIMARY, delta=None):
    a = df[df.name == a_name].set_index("seed")[metric]
    b = df[df.name == b_name].set_index("seed")[metric]
    common = a.index.intersection(b.index)
    if len(common) < 2:
        return {"n": len(common), "diff": np.nan, "lo": np.nan, "hi": np.nan, "p": np.nan, "p_ni": np.nan}
    d = (a[common] - b[common]).to_numpy(float)
    rng = np.random.default_rng(0)
    boots = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(1)
    p = 1.0 if np.allclose(d, 0) else float(wilcoxon(d).pvalue)
    out = {"n": len(common), "diff": float(d.mean()), "lo": float(np.percentile(boots, 2.5)),
           "hi": float(np.percentile(boots, 97.5)), "p": p}
    if delta is not None:
        x = d + delta
        out["p_ni"] = 1.0 if np.allclose(x, 0) else float(wilcoxon(x, alternative="greater").pvalue)
    return out


def holm(ps):
    ps = np.asarray(ps, float); order = np.argsort(ps); m = len(ps)
    adj = np.empty(m); run = 0.0
    for i, j in enumerate(order):
        run = max(run, min(1.0, (m - i) * ps[j]))
        adj[j] = run
    return adj


def agg(df, by="name", cols=None):
    cols = [c for c in (cols or []) if c in df]
    g = df.groupby(by)[cols]
    out = g.mean().add_suffix("_mean").join(g.std().add_suffix("_sd")).join(df.groupby(by).size().rename("n_seeds"))
    return out


def best_uniform(df, max_mb):
    """Uniform period chosen on VALIDATION among those with mean upload <= max_mb."""
    cand = df[df.name.map(lambda n: _uniform_k(n) is not None)]
    if cand.empty:
        return None
    m = cand.groupby("name")[["total_upload_mb", "fed_val_auroc_valsel"]].mean()
    m = m[m.total_upload_mb <= max_mb]
    return None if m.empty else m.fed_val_auroc_valsel.idxmax()


# --------------------------------------------------------------------------- #
def main_table(df):
    sp = df[(df.setting == "specialty") & df.eps.isna()]
    cols = [PRIMARY, "valsel_round", "upload_mb_valsel", "total_upload_mb", "rounds_transmitting",
            "fed_auroc_final", "fed_auroc_last10_mean", "fed_auroc_phase", "fed_ft1_auroc_final",
            "fed_ft3_auroc_final", "fed_auprc_valsel", "fed_ece_valsel", "fed_cal_ece_valsel",
            "fed_cal_brier_valsel", "fed_cal_slope_valsel", "fed_auroc_site_min_valsel",
            "fed_auroc_site_p10_valsel", "n_clients_single_class_valsel", "peak_frac", "realised_payload_fraction"]
    t = agg(sp[sp.name.isin(MAIN_ORDER + ["central", "hgb"])], cols=cols)
    for metric, lab in ((PRIMARY, "valsel"), ("fed_auroc_last10_mean", "last10"),
                        ("fed_auroc_phase", "phase"), ("fed_auroc_final", "final")):
        for n in t.index:
            r = paired(sp, n, "fedper", metric, delta=0.01 if lab == "valsel" else None)
            t.loc[n, f"d_fedper_{lab}"] = r["diff"]; t.loc[n, f"p_fedper_{lab}"] = r["p"]
            if lab == "valsel":
                t.loc[n, "ci_lo_fedper_valsel"], t.loc[n, "ci_hi_fedper_valsel"] = r["lo"], r["hi"]
                t.loc[n, "p_ni010_fedper_valsel"] = r.get("p_ni", np.nan)
                t.loc[n, "p_ni005_fedper_valsel"] = paired(sp, n, "fedper", metric, delta=0.005).get("p_ni", np.nan)
    order = [n for n in MAIN_ORDER + ["central", "hgb"] if n in t.index]
    return t.loc[order]


def frontier_tables(df):
    sp = df[(df.setting == "specialty") & df.eps.isna() & ~df.name.isin(["central", "hgb"])].copy()
    sp["family"] = sp.name.map(family)
    fr = agg(sp, by=["family", "name"], cols=[PRIMARY, "fed_val_auroc_valsel", "total_upload_mb",
                                               "upload_mb_valsel", "fed_auroc_last10_mean", "fed_auroc_phase"])
    rows = []
    for fam in sorted(sp.family.unique()):
        f = sp[sp.family == fam]
        m = f.groupby("name")[["total_upload_mb", "fed_val_auroc_valsel", PRIMARY]].mean()
        for B in BUDGETS:
            ok = m[m.total_upload_mb <= B]
            if ok.empty:                       # truncated at the family's own budget: no extrapolation
                rows.append({"family": fam, "budget_mb": B, "config": None})
                continue
            n = ok.fed_val_auroc_valsel.idxmax()
            v = f[f.name == n][PRIMARY]
            rows.append({"family": fam, "budget_mb": B, "config": n, "mb": float(ok.loc[n, "total_upload_mb"]),
                         "auroc_mean": float(v.mean()), "auroc_sd": float(v.std()), "n_seeds": int(v.size)})
    return fr, pd.DataFrame(rows)


def gate_table(df):
    rows = []
    settings = (["specialty"] + sorted([s for s in df.setting.unique() if s.startswith("a")],
                                       key=lambda s: float(s[1:]))
                + [s for s in ("resnet", "transformer") if s in set(df.setting)])
    for st in settings:
        d = df[(df.setting == st) & df.eps.isna()]
        if "tierfed" not in set(d.name):
            continue
        tf_mb = d[d.name == "tierfed"].total_upload_mb.mean()
        bu = best_uniform(d, MATCH_TOL * tf_mb)
        r_u = paired(d, "tierfed", bu) if bu else {"diff": np.nan, "lo": np.nan, "hi": np.nan, "p": np.nan}
        r_p = paired(d, "tierfed", "fedper")
        row = {"setting": st, "tierfed_auroc": d[d.name == "tierfed"][PRIMARY].mean(), "tierfed_mb": tf_mb,
               "best_uniform": bu,
               "best_uniform_auroc": d[d.name == bu][PRIMARY].mean() if bu else np.nan,
               "best_uniform_mb": d[d.name == bu].total_upload_mb.mean() if bu else np.nan,
               "gain_vs_uniform": r_u["diff"], "gain_lo": r_u["lo"], "gain_hi": r_u["hi"], "p_vs_uniform": r_u["p"],
               "gain_vs_fedper": r_p["diff"], "p_vs_fedper": r_p["p"],
               "fedper_auroc": d[d.name == "fedper"][PRIMARY].mean(),
               "fedavg_auroc": d[d.name == "fedavg"][PRIMARY].mean(),
               "local_auroc": d[d.name == "local"][PRIMARY].mean(),
               "central_auroc": d[d.name == "central"][PRIMARY].mean(),
               "hgb_auroc": d[d.name == "hgb"][PRIMARY].mean()}
        for n in ("tierfed", "tierfed_db", "tierfed_st"):
            x = d[d.name == n]
            if "sch_frac_rounds_tier_differentiated" in x:
                row[f"diff_rate_{n}"] = x["sch_frac_rounds_tier_differentiated"].mean()
        for n in ("tierfed_db", "tierfed_st", "tierfed_rho0.15", "tierfed_rho0.4"):
            if n in set(d.name):
                bu_n = best_uniform(d, MATCH_TOL * d[d.name == n].total_upload_mb.mean())
                r = paired(d, n, bu_n) if bu_n else {"diff": np.nan, "p": np.nan}
                row[f"gain_{n}_vs_uniform"] = r["diff"]; row[f"p_{n}_vs_uniform"] = r["p"]
        rows.append(row)
    g = pd.DataFrame(rows)
    if not g.empty:
        ok = g.p_vs_uniform.notna()
        g.loc[ok, "p_vs_uniform_holm"] = holm(g.loc[ok, "p_vs_uniform"])
        ok = g.p_vs_fedper.notna()
        g.loc[ok, "p_vs_fedper_holm"] = holm(g.loc[ok, "p_vs_fedper"])
    return g


def peak_table(df, uplink_mbps=5.0, rtt_s=0.03, fracs=(0.6, 0.8, 1.0)):
    """Window violations on the cell-edge link for windows W = f * t_full, where
    t_full is the time to upload one site's full shared payload (M-1)."""
    sp = df[(df.setting == "specialty") & df.eps.isna() & df.name.isin(MAIN_ORDER)]
    rows = []
    for _, r in sp.iterrows():
        rb = r["_round_bits"]
        if not rb or not r.get("shared_bits"):
            continue
        t_full = r["shared_bits"] / (uplink_mbps * 1e6) + rtt_s
        times = [b["max_client_bits"] / (uplink_mbps * 1e6) + rtt_s for b in rb if b["upload_bits"] > 0]
        out = {"name": r["name"], "seed": r["seed"], "peak_frac": r["peak_frac"], "p95_frac": r["p95_frac"],
               "round_time_max_s": max(times) if times else 0.0,
               "round_time_p95_s": sorted(times)[min(len(times) - 1, int(0.95 * len(times)))] if times else 0.0,
               "t_full_s": t_full}
        for f in fracs:
            out[f"violation_rate_W{f:g}"] = sum(t > f * t_full + 1e-12 for t in times) / len(rb)
        rows.append(out)
    p = pd.DataFrame(rows)
    return p.groupby("name").mean(numeric_only=True).drop(columns=["seed"]).reindex(
        [n for n in MAIN_ORDER if n in set(p.name)]) if not p.empty else p


def _auc(pos, neg):
    """Mann-Whitney AUC: P(statistic under the shifted world > under the clean one)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return np.nan
    gt = (pos[:, None] > neg[None, :]).mean(); eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)


def timing_table(df):
    """D-8: what a passive observer of the traffic learns about one site's data.
    Observables per run: rounds with any traffic, rounds in which the target site
    (site 0) transmits, and the bytes it sends. AUC 0.5 means no information."""
    t = df[df.setting.isin(["tm_clean", "tm_shift"])]
    rows = []
    for n in sorted(t.name.unique()):
        obs = {}
        for w in ("tm_clean", "tm_shift"):
            x = t[(t.name == n) & (t.setting == w)]
            tx = [sum(1 for b in rb if b["upload_bits"] > 0) for rb in x["_round_bits"]]
            tgt = [sum(1 for b in rb if (b.get("client_bits") or [0])[0] > 0) for rb in x["_round_bits"]]
            mb = [sum((b.get("client_bits") or [0])[0] for b in rb) / 8e6 for rb in x["_round_bits"]]
            obs[w] = (tx, tgt, mb)
        if not obs["tm_clean"][0] or not obs["tm_shift"][0]:
            continue
        r = {"name": n, "n_clean": len(obs["tm_clean"][0]), "n_shift": len(obs["tm_shift"][0])}
        for i, lab in enumerate(("tx_rounds", "target_tx_rounds", "target_mb")):
            a = _auc(obs["tm_shift"][i], obs["tm_clean"][i])
            r[f"auc_{lab}"] = a
            r[f"adv_{lab}"] = abs(a - 0.5) * 2 if np.isfinite(a) else np.nan   # 0 = no leak, 1 = perfect
            r[f"mean_clean_{lab}"] = float(np.mean(obs["tm_clean"][i]))
            r[f"mean_shift_{lab}"] = float(np.mean(obs["tm_shift"][i]))
        rows.append(r)
    return pd.DataFrame(rows)


def dp_table(df):
    d = df[df.eps.notna()]
    if d.empty:
        return d
    cols = [PRIMARY, "fed_auroc_final", "fed_auroc_last10_mean", "dp_rounds_charged", "dp_sigma",
            "dp_epsilon", "clip_fraction", "total_upload_mb"]
    t = agg(d, by=["q", "eps", "name"], cols=cols).reset_index()
    loc = df[(df.setting == "specialty") & (df.name == "local") & df.eps.isna()][PRIMARY]
    t.attrs["local_auroc"] = (float(loc.mean()), float(loc.std())) if len(loc) else (np.nan, np.nan)
    return t


# --------------------------------------------------------------------------- #
def fmt(x, nd=4):
    return "" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def report(out: Path, main, fr, budget, gate, peak, dp, derma):
    L = ["# v6 audit report", "",
         "Primary metric: test AUROC at the round chosen on per-site validation (patient-grouped 70/10/20).", ""]
    if not gate.empty:
        L += ["## Decision gate (TierFed rho = 0.25 against the best uniform period at matched bytes)", "",
              "| setting | TierFed | MB | best uniform | its AUROC | MB | gain [95% CI] | p | p Holm | diff. rate |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in gate.iterrows():
            L.append(f"| {r.setting} | {fmt(r.tierfed_auroc)} | {fmt(r.tierfed_mb, 1)} | {r.best_uniform} | "
                     f"{fmt(r.best_uniform_auroc)} | {fmt(r.best_uniform_mb, 1)} | {fmt(r.gain_vs_uniform)} "
                     f"[{fmt(r.gain_lo)}, {fmt(r.gain_hi)}] | {fmt(r.p_vs_uniform, 3)} | "
                     f"{fmt(r.get('p_vs_uniform_holm', np.nan), 3)} | {fmt(r.get('diff_rate_tierfed', np.nan), 2)} |")
        sig = gate[(gate.get("p_vs_uniform_holm", pd.Series(dtype=float)) < 0.05) &
                   (gate.gain_vs_uniform >= PRACTICAL)]
        route = "A" if len(sig) else "B"
        L += ["", f"Route {route}: TierFed beats the best uniform period at matched bytes, Holm p < 0.05 and "
              f"gain >= {PRACTICAL}, in {len(sig)} of {len(gate)} settings.", ""]
        (out / "gate.json").write_text(json.dumps({"route": route, "settings_won": sig.setting.tolist(),
                                                   "rows": json.loads(gate.to_json(orient="records"))}, indent=1))
    if not main.empty:
        L += ["## Main table (specialty sites)", "",
              "| method | val-sel. AUROC | round | MB to sel. | MB | last-10 | phase | final | +FT1 | +FT3 | "
              "ECE | ECE cal. | worst site | d vs FedPer | p |", "|" + "---|" * 15]
        for n, r in main.iterrows():
            g = lambda k, nd=4: fmt(r.get(f"{k}_mean", np.nan), nd)  # noqa: E731
            L.append(f"| {n} | {g(PRIMARY)} | {g('valsel_round', 1)} | {g('upload_mb_valsel', 1)} | "
                     f"{g('total_upload_mb', 1)} | {g('fed_auroc_last10_mean')} | {g('fed_auroc_phase')} | "
                     f"{g('fed_auroc_final')} | {g('fed_ft1_auroc_final')} | {g('fed_ft3_auroc_final')} | "
                     f"{g('fed_ece_valsel', 3)} | {g('fed_cal_ece_valsel', 3)} | {g('fed_auroc_site_min_valsel')} | "
                     f"{fmt(r.get('d_fedper_valsel', np.nan))} | {fmt(r.get('p_fedper_valsel', np.nan), 3)} |")
        L.append("")
    if not budget.empty:
        L += ["## Accuracy at a byte budget (configuration chosen on validation, no extrapolation)", "",
              "| family | 30 MB | 60 MB | 120 MB |", "|---|---|---|---|"]
        for fam, g in budget.groupby("family"):
            cells = []
            for B in BUDGETS:
                x = g[g.budget_mb == B]
                cells.append("" if x.empty or x.config.isna().all() else
                             f"{fmt(float(x.auroc_mean.iloc[0]))} ({x.config.iloc[0]})")
            L.append(f"| {fam} | " + " | ".join(cells) + " |")
        L.append("")
    if not peak.empty:
        L += ["## Peak load on the 5 Mbit/s link (M-1)", "",
              "| method | peak / full payload | p95 | violations at W = 0.6 t_full | at 0.8 t_full |",
              "|---|---|---|---|---|"]
        for n, r in peak.iterrows():
            L.append(f"| {n} | {fmt(r.peak_frac, 3)} | {fmt(r.p95_frac, 3)} | {fmt(r['violation_rate_W0.6'], 3)} | "
                     f"{fmt(r['violation_rate_W0.8'], 3)} |")
        L.append("")
    if dp is not None and not dp.empty:
        la = dp.attrs.get("local_auroc", (np.nan, np.nan))
        L += ["## Matched epsilon (q = 1 primary)", "", f"Local-only (epsilon = 0): {fmt(la[0])} +- {fmt(la[1])}", "",
              "| q | eps | method | AUROC | releases | sigma | clipped |", "|---|---|---|---|---|---|---|"]
        for _, r in dp.sort_values(["q", "eps", "name"]).iterrows():
            L.append(f"| {r.q:g} | {r.eps:g} | {r['name']} | {fmt(r.get(PRIMARY + '_mean'))} | "
                     f"{fmt(r.get('dp_rounds_charged_mean'), 0)} | {fmt(r.get('dp_sigma_mean'), 2)} | "
                     f"{fmt(r.get('clip_fraction_mean'), 2)} |")
        L.append("")
    if derma is not None and not derma.empty:
        L += ["## DermaMNIST, malignant vs benign, alpha = 0.1", "", "| method | AUROC | MB |", "|---|---|---|"]
        for n, r in derma.iterrows():
            L.append(f"| {n} | {fmt(r.get(PRIMARY + '_mean'))} | {fmt(r.get('total_upload_mb_mean'), 1)} |")
    (out / "REPORT.md").write_text("\n".join(L) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--dataset", default="diabetes")
    a = ap.parse_args(argv)
    out = Path(a.results) / "tables" / "audit"; out.mkdir(parents=True, exist_ok=True)
    df = load(a.results, a.dataset)
    if df.empty:
        print("no _au_ logs found"); return
    tm = timing_table(df)
    if not tm.empty:
        tm.to_csv(out / "timing.csv", index=False)
    base = df[(df.setting != "derma") & ~df.setting.str.startswith("tm_")]
    main_t = main_table(base); main_t.to_csv(out / "main.csv")
    fr, budget = frontier_tables(base); fr.to_csv(out / "frontier.csv"); budget.to_csv(out / "budget.csv", index=False)
    gate = gate_table(base); gate.to_csv(out / "gate.csv", index=False)
    peak = peak_table(base); peak.to_csv(out / "peak.csv")
    dp = dp_table(base)
    if dp is not None and not dp.empty:
        dp.to_csv(out / "dp.csv", index=False)
    der = df[df.setting == "derma"]
    derma = agg(der, cols=[PRIMARY, "total_upload_mb", "fed_auroc_last10_mean"]) if not der.empty else None
    if derma is not None:
        derma.to_csv(out / "derma.csv")
    report(out, main_t, fr, budget, gate, peak, dp, derma)
    if not tm.empty:
        with open(out / "REPORT.md", "a") as fh:
            fh.write("\n## Timing side channel (D-8): AUC of the observer, shifted vs clean target site\n\n"
                     "| method | rounds with traffic | target's rounds | target's MB |\n|---|---|---|---|\n")
            for _, r in tm.iterrows():
                fh.write(f"| {r['name']} | {fmt(r.auc_tx_rounds, 2)} | {fmt(r.auc_target_tx_rounds, 2)} | "
                         f"{fmt(r.auc_target_mb, 2)} |\n")
    print((out / "REPORT.md").read_text())


if __name__ == "__main__":
    main()

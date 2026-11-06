"""Build every table of the paper from the per-seed logs on disk.

Outputs
  results/tables/*.csv            human-readable summaries and tests
  paper/generated/*.tex           LaTeX table bodies the manuscript \\inputs
  paper/generated/numbers.tex     \\newcommand macros for every number used in prose

No number in the manuscript is typed by hand. A table whose runs do not
exist renders the string RUN REQUIRED so that `make paper` cannot silently
produce a manuscript with missing evidence.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .stats import compare_all, non_inferiority, summarise_runs

METRICS = ["fed_auroc_final", "fed_auroc_best", "fed_auroc_earlystop_value", "fed_auroc_last10_mean",
           "fed_auprc_final", "fed_auprc_best", "fed_f1_best_final", "fed_ece_final", "fed_brier_final",
           "pooled_auroc_final", "pooled_auprc_final", "global_auroc_final",
           "total_upload_mb", "rounds_transmitting", "realised_payload_fraction",
           "divergence_total_mean", "mia_auc_mean", "inversion_nmse_mean", "inversion_feature_corr_mean",
           "n_params_exposed_mean", "wall_s", "s_per_round", "peak_mem_mb"]
NET = ["net_nr_cell_edge_comm_time_s", "net_nr_typical_comm_time_s", "net_sixg_slice_comm_time_s",
       "net_leo_sat_comm_time_s", "net_hospital_wan_comm_time_s",
       "net_nr_cell_edge_energy_j", "net_leo_sat_energy_j", "net_nr_cell_edge_window_compliance"]
NAMES = {"fedavg": "FedAvg", "fedprox": "FedProx", "scaffold": "SCAFFOLD", "fedper": "FedPer",
         "fedlama": "FedLAMA", "adacomm": "AdaComm", "lag": "LAG", "fixed": "Fixed schedule",
         "tierfed": "TierFed", "tierfed_dp": "TierFed-DP"}
RUN_REQUIRED = r"\textbf{[RUN REQUIRED]} \\"


# --------------------------------------------------------------------------- #
def load_summaries(log_dir) -> pd.DataFrame:
    rows: List[Dict] = []
    for p in sorted(Path(log_dir).glob("*_summary.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        cfg = d.get("config", {})
        row = {"file": p.name, "dataset": cfg.get("dataset"), "model": cfg.get("model"),
               "strategy": cfg.get("strategy"), "seed": cfg.get("seed"), "rounds": cfg.get("rounds"),
               "n_clients": (d.get("dataset") or {}).get("n_clients"), "loss": cfg.get("loss"),
               "compressor": cfg.get("compressor") or "none", "tag": cfg.get("tag", "") or ""}
        sk = cfg.get("strategy_kwargs") or {}
        row["rho"] = sk.get("rho"); row["warmup"] = sk.get("warmup"); row["solver"] = sk.get("solver", "exact")
        row["k_slow"] = sk.get("k_slow"); row["k_med"] = sk.get("k_med")
        row["private_head"] = sk.get("private_head"); row["schedule"] = sk.get("schedule")
        dp = d.get("dp") or {}
        row["dp_sigma"] = dp.get("dp_sigma", 0.0) if dp.get("dp_enabled") else 0.0
        row["dp_epsilon"] = dp.get("dp_epsilon") if dp.get("dp_enabled") else None
        row["dp_rounds_charged"] = dp.get("dp_rounds_charged")
        row["dp_q"] = dp.get("dp_sample_rate", 1.0) if dp.get("dp_enabled") else None
        row["dp_target_eps"] = dp.get("dp_target_epsilon") if dp.get("dp_enabled") else None
        row["lam"] = sk.get("lam")
        tr = d.get("tier_report") or {}
        for t in ("slow", "medium", "fast", "total"):
            if t in tr:
                row[f"tier_{t}_params"] = tr[t]["params"]; row[f"tier_{t}_pct"] = tr[t]["pct"]
        ds = d.get("dataset") or {}
        for k in ("n_train", "n_test", "n_features", "n_patients_train", "prevalence_train", "deduplicated"):
            row[f"ds_{k}"] = ds.get(k)
        pr = ds.get("partition_report") or {}
        for k in ("size_max_share", "size_gini", "label_tv_mean", "n_single_class_clients", "size_min", "size_max"):
            row[f"part_{k}"] = pr.get(k)
        sch = d.get("schedule") or {}
        for k in ("K_slow", "K_medium", "K_slow_mean", "K_medium_mean", "n_distinct_schedules",
                  "frac_rounds_tier_differentiated", "gamma_slow", "gamma_medium", "w_slow", "w_medium",
                  "mean_interval", "tau_final", "planned_sigma", "planned_releases"):
            if k in sch:
                row[f"sched_{k}"] = sch[k]
        row.update(d.get("summary", {}))
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def _f(x, nd=4):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "--"
        return f"{float(x):.{nd}f}"
    except Exception:
        return "--"


def _p(x):
    try:
        x = float(x)
    except Exception:
        return "--"
    if np.isnan(x):
        return "--"
    if x < 1e-4:
        return r"\ensuremath{<10^{-4}}"
    return r"\ensuremath{<0.001}" if x < 1e-3 else f"{x:.3f}"


_DIG = dict(zip("0123456789", ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]))


def _safe(name: str) -> str:
    """LaTeX command names allow letters only: spell digits out, drop the rest."""
    return re.sub(r"[^A-Za-z]", "", re.sub(r"\d", lambda m: _DIG[m.group()], name))


def _macro(name: str, value: str) -> str:
    name = _safe(name)
    return f"\\newcommand{{\\{name}}}{{{value}}}\n"


class TexWriter:
    def __init__(self, gen_dir: Path):
        self.dir = gen_dir; self.dir.mkdir(parents=True, exist_ok=True)
        self.numbers: List[str] = []
        self.flags: Dict[str, str] = {}

    def table(self, name: str, body: Optional[str]):
        (self.dir / f"{name}.tex").write_text(body if body else RUN_REQUIRED + "\n")

    def num(self, name: str, value):
        # p-value macros (names ending in P) carry their own relation so prose can
        # write $p\num{...}$ and get "p=0.036" or "p<10^{-4}"
        if name.endswith("P") and isinstance(value, str) and value[:1].isdigit():
            value = "=" + value
        self.numbers.append(_macro(name, value))

    def flag(self, name: str, value: bool):
        self.flags[_safe(name)] = "true" if value else "false"

    def finish(self):
        txt = "".join(self.numbers)
        txt += "".join(_macro("flag" + k, v) for k, v in self.flags.items())
        (self.dir / "numbers.tex").write_text(txt)
        (self.dir / "flags.json").write_text(json.dumps(self.flags, indent=2))


# --------------------------------------------------------------------------- #
def main_benchmark(df: pd.DataFrame, T: Path, W: TexWriter, dataset: str, model: str):
    if df.empty:
        W.table(f"main_{dataset}_{model}", None); W.table(f"holm_{dataset}_{model}", None)
        W.table(f"network_{dataset}_{model}", None); return
    for col, default in (("dp_sigma", 0.0), ("compressor", "none"), ("tag", "")):
        if col not in df:
            df[col] = default
    sub = df[(df.dataset == dataset) & (df.model == model) & (df.tag.fillna("") == "")
             & (df["dp_sigma"].fillna(0) == 0) & (df["compressor"].fillna("none") == "none")]
    sub = sub.drop_duplicates(subset=["strategy", "seed"], keep="last")
    key = f"{dataset}_{model}"
    if sub.empty:
        W.table(f"main_{key}", None); return
    metrics = [m for m in METRICS + NET if m in sub]
    summ = summarise_runs(sub, ["strategy"], metrics)
    summ.to_csv(T / f"benchmark_{key}_summary.csv", index=False)
    tests = compare_all(sub, "strategy", "fedper", [m for m in ("fed_auroc_final", "fed_auprc_final") if m in sub])
    tests.to_csv(T / f"benchmark_{key}_tests.csv", index=False)
    ni = {}
    fp = sub[sub.strategy == "fedper"].set_index("seed")
    for s in sorted(sub.strategy.unique()):
        if s == "fedper":
            continue
        cur = sub[sub.strategy == s].set_index("seed")
        common = sorted(set(cur.index) & set(fp.index))
        if len(common) < 2:
            continue
        for m, margin in (("fed_auroc_final", 0.01), ("fed_auprc_final", 0.01)):
            if m in sub:
                ni[f"{s}|{m}"] = non_inferiority(cur.loc[common, m].values, fp.loc[common, m].values, margin)
    (T / f"benchmark_{key}_noninferiority.json").write_text(json.dumps(ni, indent=2))

    order = [s for s in NAMES if s in set(summ.strategy)]
    S = summ.set_index("strategy")
    lines = []
    for s in order:
        r = S.loc[s]
        ni_a = ni.get(f"{s}|fed_auroc_final", {}); ni_p = ni.get(f"{s}|fed_auprc_final", {})
        lines.append(" & ".join([
            NAMES[s],
            f"{_f(r.get('fed_auroc_final_mean'))} [{_f(r.get('fed_auroc_final_ci_lo'))}, {_f(r.get('fed_auroc_final_ci_hi'))}]",
            _f(r.get("fed_auprc_final_mean"), 3),
            _f(r.get("fed_auroc_earlystop_value_mean")),
            _f(r.get("pooled_auroc_final_mean")),
            _f(r.get("total_upload_mb_mean"), 1),
            _f(r.get("rounds_transmitting_mean"), 0),
            "(ref.)" if s == "fedper" else _p(ni_a.get("p")),
            "(ref.)" if s == "fedper" else _p(ni_p.get("p")),
        ]) + r" \\")
    best = max(_lead_float(l.split(" & ")[1]) for l in lines)
    lines = [" & ".join([c if (i != 1 or _lead_float(c) != best) else _bold(c) for i, c in enumerate(l.split(" & "))]) for l in lines]
    W.table(f"main_{key}", "\n".join(lines) + "\n")
    # single-column variant: method, AUROC [CI], AUPRC, upload, tx rounds, NI_AUROC p
    short = [" & ".join([c.strip() for i, c in enumerate(l.replace(" \\\\", "").split(" & ")) if i in (0, 1, 2, 5, 6, 7)]) + " \\\\" for l in lines]
    W.table(f"mainshort_{key}", "\n".join(short) + "\n")

    # Holm-corrected paired differences table
    tl = []
    tests_a = tests[tests.metric == "fed_auroc_final"] if not tests.empty else tests
    for _, r in tests_a.iterrows():
        if "p_holm" not in r or r.get("n_seeds", 0) < 2:
            continue
        tl.append(" & ".join([f"{NAMES.get(r['method'], r['method'])} vs FedPer",
                              f"{r['mean_diff']:+.4f}", f"[{_f(r.get('diff_ci_lo'))}, {_f(r.get('diff_ci_hi'))}]",
                              _p(r.get("p_holm")), _f(r.get("hedges_g"), 2),
                              r"\checkmark" if r.get("significant_holm") else ""]) + r" \\")
    W.table(f"holm_{key}", "\n".join(tl) + "\n" if tl else None)

    # prose macros
    for s in order:
        r = S.loc[s]; tagn = re.sub(r"[^A-Za-z]", "", s).capitalize()
        W.num(f"{key}{tagn}Auroc", _f(r.get("fed_auroc_final_mean")))
        W.num(f"{key}{tagn}Auprc", _f(r.get("fed_auprc_final_mean"), 3))
        W.num(f"{key}{tagn}UploadMb", _f(r.get("total_upload_mb_mean"), 1))
        W.num(f"{key}{tagn}Seeds", str(int(r["n_seeds"])))
        if s != "fedper":
            a = ni.get(f"{s}|fed_auroc_final", {}); b = ni.get(f"{s}|fed_auprc_final", {})
            W.num(f"{key}{tagn}VsFedperDiff", f"{a.get('mean_diff', float('nan')):+.4f}")
            W.num(f"{key}{tagn}NiAurocP", _p(a.get("p"))); W.num(f"{key}{tagn}NiAuprcP", _p(b.get("p")))
            W.flag(f"{key}{tagn}NonInferiorAuroc", bool(a.get("non_inferior", False)))
            W.flag(f"{key}{tagn}NonInferiorAuprc", bool(b.get("non_inferior", False)))
            fpm = S.loc["fedper"]["total_upload_mb_mean"] if "fedper" in S.index else np.nan
            W.num(f"{key}{tagn}UploadSavingPct", _f(100 * (1 - r.get("total_upload_mb_mean") / fpm), 0) if fpm else "--")
    # dataset facts (from the runs themselves)
    r0 = sub.iloc[0]
    W.num(f"{key}NTrain", f"{int(r0.get('ds_n_train', 0)):,}")
    W.num(f"{key}NTest", f"{int(r0.get('ds_n_test', 0)):,}")
    W.num(f"{key}NFeatures", str(int(r0.get("ds_n_features", 0))))
    W.num(f"{key}NPatients", f"{int(r0.get('ds_n_patients_train', 0)):,}")
    W.num(f"{key}Prevalence", _f(100 * r0.get("ds_prevalence_train", np.nan), 1))
    W.num(f"{key}PrevalenceFrac", _f(r0.get("ds_prevalence_train", np.nan), 2))   # the AUPRC baseline
    W.num(f"{key}MaxClientShare", _f(100 * r0.get("part_size_max_share", np.nan), 0))
    W.num(f"{key}LabelTv", _f(r0.get("part_label_tv_mean", np.nan), 3))
    for t in ("slow", "medium", "fast", "total"):
        W.num(f"{key}Tier{t.capitalize()}Params", f"{int(r0.get(f'tier_{t}_params', 0)):,}")
        W.num(f"{key}Tier{t.capitalize()}Pct", _f(r0.get(f"tier_{t}_pct", np.nan), 1))

    # network table on the same dataset
    nl = []
    for s in order:
        r = S.loc[s]
        nl.append(" & ".join([NAMES[s], _f(r.get("total_upload_mb_mean"), 1),
                              _f(r.get("net_nr_cell_edge_comm_time_s_mean"), 0),
                              _f(r.get("net_leo_sat_comm_time_s_mean"), 0),
                              _f(r.get("net_sixg_slice_comm_time_s_mean"), 1),
                              _f(r.get("net_nr_cell_edge_energy_j_mean"), 1),
                              _f(100 * r.get("net_nr_cell_edge_window_compliance_mean", np.nan), 0)]) + r" \\")
    W.table(f"network_{key}", "\n".join(nl) + "\n")


def _lead_float(cell: str):
    m = re.search(r"-?\d+\.\d+|-?\d+", cell.replace("\\textbf{", ""))
    return float(m.group()) if m else float("nan")


def _bold(cell: str) -> str:
    """Bold the leading number of a cell ("0.6269 $\\pm$ 0.0082" -> "\\textbf{0.6269} $\\pm$ 0.0082")."""
    m = re.search(r"-?\d+\.\d+|-?\d+", cell)
    return cell if not m else cell[:m.start()] + "\\textbf{" + m.group() + "}" + cell[m.end():]


def family(df, T, W, name, mask, keys, cols, label_fn, metrics=None, bold=None, rule_on=None, higher=True):
    """One generated table body. `bold` is the index (into cols) of the column
    whose best value is set in bold, within each group given by `rule_on` (a
    function of the summary row); `rule_on` also inserts a rule between groups."""
    sub = df[mask]
    if sub.empty:
        W.table(name, None); return None
    metrics = metrics or [m for m in METRICS + NET if m in sub]
    out = summarise_runs(sub, keys, metrics)
    out.to_csv(T / f"ablation_{name}.csv", index=False)
    rows = [(label_fn(r), [c(r) for c in cols], rule_on(r) if rule_on else 0) for _, r in out.iterrows()]
    if bold is not None:
        groups = {}
        for i, (_, cells, g) in enumerate(rows):
            groups.setdefault(g, []).append((i, _lead_float(cells[bold])))
        for g, items in groups.items():
            vals = [v for _, v in items if v == v]
            if not vals:
                continue
            best = max(vals) if higher else min(vals)
            for i, v in items:
                if v == best:
                    rows[i][1][bold] = _bold(rows[i][1][bold])
    lines, prev = [], None
    for label, cells, g in rows:
        if rule_on is not None and prev is not None and g != prev:
            lines.append(r"\midrule")
        prev = g
        lines.append(" & ".join([label] + cells) + r" \\")
    W.table(name, "\n".join(lines) + "\n")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Aggregate finished runs into paper tables")
    ap.add_argument("--results", default="results")
    ap.add_argument("--paper", default="paper/generated")
    ap.add_argument("--dataset", default="diabetes")
    ap.add_argument("--model", default="mlp")
    a = ap.parse_args(argv)

    R = Path(a.results); T = R / "tables"; T.mkdir(parents=True, exist_ok=True)
    W = TexWriter(Path(a.paper))
    df = load_summaries(R / "logs")
    if df.empty:
        print("no *_summary.json found; writing RUN REQUIRED placeholders")
        df = pd.DataFrame(columns=["dataset", "model", "tag", "strategy", "seed", "dp_sigma", "compressor",
                                   "rho", "solver", "k_slow", "k_med", "loss", "warmup", "n_clients"])
    df.to_csv(T / "collected_all_raw.csv", index=False)
    W.num("TotalRuns", str(len(df)))

    for ds, model in [(a.dataset, a.model), ("eicu", "mlp"), ("medmnist", "cnn")]:
        main_benchmark(df, T, W, ds, model)

    tag = df["tag"].fillna("")
    au = lambda r: f"{_f(r.get('fed_auroc_final_mean'))} $\\pm$ {_f(r.get('fed_auroc_final_std'))}"  # noqa: E731
    pr = lambda r: _f(r.get("fed_auprc_final_mean"), 3)      # noqa: E731
    up = lambda r: _f(r.get("total_upload_mb_mean"), 1)      # noqa: E731
    ns = lambda r: str(int(r["n_seeds"]))                    # noqa: E731

    # budget sweep
    family(df, T, W, "budget", tag.str.startswith("_budget_") & (df.get("solver", "exact") == "exact"),
           ["rho"], [au, pr, up,
                     lambda r: _f(r.get("realised_payload_fraction_mean"), 2),
                     lambda r: _f(r.get("sched_K_slow_mean_mean"), 1),
                     lambda r: _f(r.get("sched_K_medium_mean_mean"), 1),
                     lambda r: _f(100 * r.get("sched_frac_rounds_tier_differentiated_mean", np.nan), 0)],
           lambda r: _f(r["rho"], 2), bold=0,
           metrics=[m for m in METRICS + NET if m in df] + ["realised_payload_fraction", "sched_K_slow_mean",
                                                             "sched_K_medium_mean", "sched_frac_rounds_tier_differentiated"])
    family(df, T, W, "solver", tag.str.startswith("_budget_"), ["rho", "solver"], [au, up],
           lambda r: f"$\\rho={_f(r['rho'], 2)}$, {r['solver']}", bold=0, rule_on=lambda r: r["rho"])
    family(df, T, W, "baselines", tag.str.startswith("_baselines_"), ["tag"],
           [au, pr, up, lambda r: _f(r.get("rounds_transmitting_mean"), 0)],
           lambda r: {"adacomm": "AdaComm, $\\tau_0={}$", "lag": "LAG, $\\xi={}$"}.get(r["tag"].split("_")[2], "{}").format(
               re.sub(r"[a-z]+", "", r["tag"].split("_")[3]) if len(r["tag"].split("_")) > 3 else "")
           if r["tag"].split("_")[2] in ("adacomm", "lag") else NAMES.get(r["tag"].split("_")[2], r["tag"]) + ", SGD solver",
           bold=0, rule_on=lambda r: r["tag"].split("_")[2])
    _fac = {"fedavg": "FedAvg: shared head, every round", "fedper": "FedPer: private head, every round",
            "sched_only": "Fixed schedule, shared head", "fixed": "Fixed schedule, private head",
            "tierfed_nohead": "TierFed, shared head", "tierfed": "TierFed, private head"}
    family(df, T, W, "factorial", tag.str.startswith("_factorial_"), ["tag"], [au, pr, up],
           lambda r: _fac.get(r["tag"].replace("_factorial_", ""), r["tag"].replace("_factorial_", "").replace("_", " ")), bold=0)
    family(df, T, W, "periods", tag.str.startswith("_periods_"), ["k_slow", "k_med"], [au, up],
           lambda r: f"({int(r['k_slow'])}, {int(r['k_med'])})", bold=0, rule_on=lambda r: r["k_slow"])
    _arch = {"mlp": "MLP", "resnet": "ResNet", "transformer": "FT-Transformer"}
    family(df, T, W, "architecture", tag.str.startswith("_architecture_"), ["model", "strategy"],
           [au, pr, up, lambda r: f"{int(r.get('tier_total_params_mean', 0)):,}"],
           lambda r: f"{_arch.get(r['model'], r['model'])} / {NAMES.get(r['strategy'], r['strategy'])}",
           metrics=[m for m in METRICS + NET if m in df] + ["tier_total_params"], bold=0, rule_on=lambda r: r["model"])
    family(df, T, W, "loss", tag.str.startswith("_loss_"), ["loss", "strategy"], [au, pr, up],
           lambda r: f"{ {'weighted_ce': 'Inverse-frequency CE', 'focal': 'Focal', 'ldam': 'LDAM', 'cb': 'Class-balanced'}.get(r['loss'], str(r['loss']).replace('_', ' ')) }, {NAMES.get(r['strategy'], r['strategy'])}",
           bold=0, rule_on=lambda r: r["loss"])
    family(df, T, W, "warmup", tag.str.startswith("_warmup_"), ["warmup"], [au, pr, up],
           lambda r: f"TierFed, $T_w={int(r['warmup'])}$", bold=0)
    cp = df[tag.str.startswith("_comp_")]
    if cp.empty:
        W.table("compression", None)
    else:
        g = cp.groupby(["compressor", "strategy"]).agg(a=("fed_auroc_final", "mean"), s=("fed_auroc_final", "std"),
                                                       u=("total_upload_mb", "mean")).reset_index()
        cname = {"none": "None (float32)", "quant8": "8-bit quantisation", "quant4": "4-bit quantisation",
                 "topk": "Top-$k$ (1\\%)", "stc": "Sparse ternary (1\\%)", "sketch": "Count sketch"}
        lines = []
        for c in ["none", "quant8", "quant4", "topk", "stc", "sketch"]:
            gc = g[g.compressor == c].set_index("strategy")
            if gc.empty:
                continue
            cells, best = [], gc["a"].max()
            for st in ["fedavg", "fedper", "tierfed"]:
                if st in gc.index:
                    r = gc.loc[st]; aval = f"{r['a']:.4f}"
                    cells += [("\\textbf{" + aval + "}") if r["a"] == best else aval, f"{r['u']:.1f}"]
                else:
                    cells += ["--", "--"]
            lines.append(" & ".join([cname.get(c, c)] + cells) + r" \\")
        W.table("compression", "\n".join(lines) + "\n")
    legacy_dp = tag.str.startswith("_dp") & ~tag.str.startswith("_dpe")
    family(df, T, W, "dp", legacy_dp, ["strategy", "dp_sigma"],
           [lambda r: _f(r.get("dp_epsilon_mean"), 1), lambda r: _f(r.get("dp_rounds_charged_mean"), 0), au, pr],
           lambda r: f"{NAMES.get(r['strategy'], r['strategy'])} / {_f(r['dp_sigma'], 1)}",
           metrics=[m for m in METRICS if m in df] + ["dp_epsilon", "dp_rounds_charged"])
    family(df, T, W, "attack", tag == "_attack", ["strategy"],
           [lambda r: f"{int(r.get('n_params_exposed_mean_mean', 0)):,}",
            lambda r: _f(r.get("inversion_nmse_mean_mean"), 2),
            lambda r: _f(r.get("inversion_feature_corr_mean_mean"), 3),
            lambda r: _f(r.get("mia_auc_mean_mean"), 3), up],
           lambda r: NAMES.get(r["strategy"], r["strategy"]))
    family(df, T, W, "scaling", tag.str.startswith("_scale"), ["n_clients", "strategy"],
           [au, up, lambda r: _f(r.get("s_per_round_mean"), 2), lambda r: _f(r.get("peak_mem_mb_mean"), 0)],
           lambda r: f"{int(r['n_clients'])} sites, {NAMES.get(r['strategy'], r['strategy'])}", bold=0, rule_on=lambda r: r["n_clients"])

    # ---------------- prose macros for the ablation families ----------------
    def _mac(prefix, sub, key_fn):
        for k, g in sub.groupby(key_fn):
            name = prefix + "".join(w[:1].upper() + w[1:] for w in str(k).replace("_", " ").split())
            W.num(name + "Auroc", _f(g["fed_auroc_final"].mean()))
            W.num(name + "Auprc", _f(g["fed_auprc_final"].mean(), 3))
            W.num(name + "UploadMb", _f(g["total_upload_mb"].mean(), 1))
            W.num(name + "TxRounds", _f(g["rounds_transmitting"].mean(), 0))
            W.num(name + "Seeds", str(len(g)))
            if "dp_epsilon" in g and g["dp_epsilon"].notna().any():
                W.num(name + "Eps", _f(g["dp_epsilon"].astype(float).mean(), 1))
    sig = {0.5: "SigmaHalf", 1.0: "SigmaOne", 2.0: "SigmaTwo", 4.0: "SigmaFour"}
    num_word = lambda x: {"0.15": "Fifteen", "0.25": "TwentyFive", "0.4": "Forty", "0.6": "Sixty", "0.8": "Eighty"}.get(str(x), str(x))  # noqa: E731
    if not df.empty:
        dpd = df[tag.str.startswith("_dp") & ~tag.str.startswith("_dpe")]
        if not dpd.empty:
            _mac("dp", dpd, lambda i: f"{dpd.loc[i,'strategy']} {sig.get(float(dpd.loc[i,'dp_sigma']), 'Sigma')}")
            for s_, sw in sig.items():
                piv = dpd[dpd.dp_sigma.astype(float) == s_].pivot_table(index="seed", columns="strategy", values="fed_auroc_final")
                if "tierfed" not in piv:
                    continue
                for o in piv.columns:
                    if o == "tierfed":
                        continue
                    d = (piv["tierfed"] - piv[o]).dropna()
                    if len(d) >= 2:
                        from scipy import stats as sps
                        p = sps.ttest_rel(piv.loc[d.index, "tierfed"], piv.loc[d.index, o]).pvalue
                        nm = "dpTfVs" + o.title() + sw
                        W.num(nm + "Diff", f"{d.mean():+.3f}"); W.num(nm + "P", _p(p))
                        W.flag(nm + "Better", bool(d.mean() > 0 and p < 0.05))
                        W.flag(nm + "Worse", bool(d.mean() < 0 and p < 0.05))
        cp = df[tag.str.startswith("_comp_")]
        if not cp.empty:
            _mac("comp", cp, lambda i: f"{cp.loc[i,'strategy']} {cp.loc[i,'tag'].replace('_comp_','')}")
        pr = df[tag.str.startswith("_periods_")]
        if not pr.empty:
            _mac("periods", pr, lambda i: pr.loc[i, "tag"].replace("_periods_", "").replace("ks", "Ks").replace("km", "Km").replace("_", " "))
        bu = df[tag.str.startswith("_budget_")]
        if not bu.empty:
            _mac("budget", bu, lambda i: "rho " + num_word(bu.loc[i, "rho"]) + (" greedy" if "greedy" in bu.loc[i, "tag"] else ""))
            for k, g in bu.groupby("tag"):
                name = "budgetRho" + num_word(g["rho"].iloc[0]) + ("Greedy" if "greedy" in k else "")
                if "sched_frac_rounds_tier_differentiated" in g:
                    W.num(name + "DiffPct", _f(100 * g["sched_frac_rounds_tier_differentiated"].mean(), 0))
        for fam in ("_baselines_", "_factorial_", "_warmup_", "_loss_", "_architecture_"):
            sub = df[tag.str.startswith(fam)]
            if not sub.empty:
                _mac(fam.strip("_"), sub, lambda i, sub=sub, fam=fam: sub.loc[i, "tag"].replace(fam, "").replace("_", " "))
        at = df[tag == "_attack"]
        if not at.empty:
            for s_, g in at.groupby("strategy"):
                nm = "attack" + s_.title()
                for col, mn in (("inversion_feature_corr_mean", "Corr"), ("mia_auc_mean", "Mia"), ("inversion_nmse_mean", "Nmse")):
                    if col in g:
                        W.num(nm + mn, _f(g[col].mean(), 3))
        # paired contrasts of TierFed against every other method on the main benchmark
        mb = df[(df.dataset == a.dataset) & (df.model == a.model) & (tag == "") & (df.get("compressor", "none").fillna("none") == "none")]
        piv = mb.pivot_table(index="seed", columns="strategy", values="fed_auroc_final")
        if "tierfed" in piv:
            from scipy import stats as sps
            for o in piv.columns:
                if o == "tierfed":
                    continue
                d = (piv["tierfed"] - piv[o]).dropna()
                if len(d) >= 2:
                    p = sps.ttest_rel(piv.loc[d.index, "tierfed"], piv.loc[d.index, o]).pvalue
                    W.num("mainTfVs" + o.title() + "Diff", f"{d.mean():+.4f}"); W.num("mainTfVs" + o.title() + "P", _p(p))
        W.num("NetCellEdgeFedper", _f(mb[mb.strategy == "fedper"]["net_nr_cell_edge_comm_time_s"].mean(), 1) if "net_nr_cell_edge_comm_time_s" in mb else "--")
        W.num("NetCellEdgeTierfed", _f(mb[mb.strategy == "tierfed"]["net_nr_cell_edge_comm_time_s"].mean(), 1) if "net_nr_cell_edge_comm_time_s" in mb else "--")

    img = df[(df["dataset"] == "medmnist") & (tag == "")] if "n_clients_single_class_final" in df else df.iloc[0:0]
    if not img.empty:
        W.num("imgSingleClassMin", str(int(img["n_clients_single_class_final"].min())))
        W.num("imgSingleClassMax", str(int(img["n_clients_single_class_final"].max())))
    v4_families(df, tag, T, W, R)
    from .theory_numbers import write as theory_numbers
    theory_numbers(R / "logs", W)
    W.finish()
    print(f"collected {len(df)} runs -> {T} and {a.paper}")
    missing = [p.name for p in Path(a.paper).glob("*.tex") if "RUN REQUIRED" in p.read_text()]
    print("tables still waiting for runs:", missing if missing else "none")


# --------------------------------------------------------------------------- #
def _paired(W, piv, a, b, name, nd=3):
    if a not in piv or b not in piv:
        return
    d = (piv[a] - piv[b]).dropna()
    if len(d) < 2:
        return
    from scipy import stats as sps
    p = sps.ttest_rel(piv.loc[d.index, a], piv.loc[d.index, b]).pvalue
    W.num(name + "Diff", f"{d.mean():+.{nd}f}"); W.num(name + "P", _p(p))
    W.flag(name + "Better", bool(d.mean() > 0 and p < 0.05))
    W.flag(name + "Worse", bool(d.mean() < 0 and p < 0.05))


_W = {1.0: "One", 2.0: "Two", 4.0: "Four", 8.0: "Eight", 0.5: "Half", 0.1: "Tenth", 10.0: "Ten", 0.3: "ThreeTenths", 3.0: "Three"}


def v4_families(df, tag, T, W, R):
    """Matched-epsilon DP, real-site benchmarks, wide model + socket emulation."""
    au = lambda r: f"{_f(r.get('fed_auroc_final_mean'))} $\\pm$ {_f(r.get('fed_auroc_final_std'))}"  # noqa: E731
    pr = lambda r: _f(r.get("fed_auprc_final_mean"), 3)      # noqa: E731
    up = lambda r: _f(r.get("total_upload_mb_mean"), 1)      # noqa: E731
    ns = lambda r: str(int(r["n_seeds"]))                    # noqa: E731
    sig = lambda r: _f(r.get("dp_sigma_mean"), 2)            # noqa: E731
    rel = lambda r: _f(r.get("dp_rounds_charged_mean"), 0)   # noqa: E731
    eps = lambda r: _f(r.get("dp_epsilon_mean"), 2)          # noqa: E731
    extra = [m for m in METRICS if m in df] + ["dp_epsilon", "dp_rounds_charged", "dp_sigma", "total_upload_mb"]
    nm = lambda s: NAMES.get(s, s)                           # noqa: E731

    # method key: fixed periods are distinguished by K
    mkey = df["strategy"].astype(str).where(df["strategy"] != "fixed",
                                            "fixed" + df["k_slow"].fillna(0).astype(float).astype(int).astype(str))
    df = df.assign(mkey=mkey)
    mlabel = {"fixed4": "Fixed $K=4$", "fixed8": "Fixed $K=8$"}
    nm2 = lambda k: mlabel.get(k, NAMES.get(k, k))                      # noqa: E731
    m = tag.str.startswith("_dpe") & ~tag.str.contains("_lam")
    _Q = {0.2: "QFifth", 0.5: "QHalf", 1.0: "QOne"}
    for N in sorted(df.loc[m, "n_clients"].dropna().unique()):
        mN = m & (df["n_clients"] == N)
        family(df, T, W, f"dpmatched{int(N)}", mN, ["dp_q", "dp_target_eps", "mkey"],
               [sig, rel, au],
               lambda r: f"{_f(r['dp_q'], 1)}, {_f(r['dp_target_eps'], 0)}, {nm2(r['mkey'])}",
               metrics=extra, bold=2, rule_on=lambda r: (r["dp_q"], r["dp_target_eps"]))
        if int(N) == 20:   # main-text body: the q = 0.5 sweep only (q = 1 rows go to the supplement)
            body = (Path(W.dir) / "dpmatched20.tex").read_text().splitlines()
            keep, outl = [], []
            for l in body:
                if l.startswith("0.5,") or (l.strip() == r"\midrule" and outl and outl[-1].strip() != r"\midrule"):
                    outl.append(l)
            while outl and outl[-1].strip() == r"\midrule":
                outl.pop()
            W.table("dpmatched20q", "\n".join(outl) + "\n")
        sub = df[mN]
        for (q, e), g in sub.groupby(["dp_q", "dp_target_eps"]):
            key = f"dpm{_safe('N' + str(int(N)))}" + _Q.get(float(q), "Q") + "Eps" + _W.get(float(e), "X")
            for s_, gs in g.groupby("mkey"):
                k2 = key + _safe(s_.title().replace("_", ""))
                W.num(k2 + "Auroc", _f(gs["fed_auroc_final"].mean()))
                W.num(k2 + "Auprc", _f(gs["fed_auprc_final"].mean(), 3))
                W.num(k2 + "Sigma", _f(gs["dp_sigma"].mean(), 2))
                W.num(k2 + "Rel", _f(gs["dp_rounds_charged"].mean(), 0))
                W.num(k2 + "UploadMb", _f(gs["total_upload_mb"].mean(), 1))
                if "sched_K_slow" in gs:
                    W.num(k2 + "K", _f(gs["sched_K_slow"].mean(), 0))
            piv = g.pivot_table(index="seed", columns="mkey", values="fed_auroc_final")
            for o in ("fedavg", "fedper", "fixed4", "fixed8"):
                _paired(W, piv, "tierfed_dp", o, key + "TfdpVs" + _safe(o.title()))
            _paired(W, piv, "fixed8", "fedper", key + "FixedEightVsFedper")
            _paired(W, piv, "fixed4", "fedper", key + "FixedFourVsFedper")
            _paired(W, piv, "fixed8", "fixed4", key + "FixedEightVsFixedFour")
    for N in sorted(df.loc[tag.str.startswith("_dpe"), "n_clients"].dropna().unique()):
        lm = tag.str.startswith("_dpe") & (df["strategy"] == "tierfed_dp") & (df["n_clients"] == N)
        lam_df = df[lm].copy()
        lam_df["lam"] = lam_df["lam"].fillna(1.0)
        # the lambda sweep is at the (q, eps) that has more than one lambda
        cnt = lam_df.groupby(["dp_q", "dp_target_eps"])["lam"].nunique()
        if cnt.empty or cnt.max() < 2:
            W.table(f"dplambda{int(N)}", None); continue
        q0, e0 = cnt.idxmax()
        lam_df = lam_df[(lam_df["dp_q"] == q0) & (lam_df["dp_target_eps"] == e0)]
        family(lam_df, T, W, f"dplambda{int(N)}", pd.Series(True, index=lam_df.index), ["lam"],
               [lambda r: _f(r.get("sched_K_slow_mean"), 0), sig, rel, au, up],
               lambda r: _f(r["lam"], 1), metrics=extra + ["sched_K_slow"], bold=3)
        W.num(f"dplam{_safe('N' + str(int(N)))}At", f"$q={q0:g}$, $\\varepsilon^\\star={e0:g}$")
        for l_, gl in lam_df.groupby("lam"):
            k2 = f"dplam{_safe('N' + str(int(N)))}" + _W.get(float(l_), _safe(str(l_)))
            W.num(k2 + "K", _f(gl["sched_K_slow"].mean(), 0)); W.num(k2 + "Auroc", _f(gl["fed_auroc_final"].mean()))
            W.num(k2 + "Rel", _f(gl["dp_rounds_charged"].mean(), 0))

    for ds in ("heart", "eicu_demo", "eicu"):
        mk = (df["dataset"] == ds) & (tag == "_real")
        fam = "real" + ds.replace("_", "")
        family(df, T, W, fam, mk, ["strategy"],
               [au, pr, up, lambda r: _f(r.get("rounds_transmitting_mean"), 0)],
               lambda r: nm(r["strategy"]), bold=0)
        g = df[mk]
        if g.empty:
            continue
        key = "real" + _safe(ds.replace("_", " ").title())
        W.num(key + "Clients", str(int(g["n_clients"].iloc[0])))
        W.num(key + "Ntrain", str(int(g["ds_n_train"].iloc[0])) if g["ds_n_train"].notna().any() else "--")
        for s_, gs in g.groupby("strategy"):
            W.num(key + s_.title() + "Auroc", _f(gs["fed_auroc_final"].mean()))
            W.num(key + s_.title() + "UploadMb", _f(gs["total_upload_mb"].mean(), 2))
        piv = g.pivot_table(index="seed", columns="strategy", values="fed_auroc_final")
        for o in piv.columns:
            if o != "tierfed":
                _paired(W, piv, "tierfed", o, key + "TfVs" + o.title())

    wide = df[tag == "_wide"]
    family(df, T, W, "wide", tag == "_wide", ["strategy"],
           [au, pr, up],
           lambda r: nm(r["strategy"]), metrics=[m for m in METRICS + NET if m in df] + ["tier_total_params"], bold=0)
    if not wide.empty:
        W.num("wideParams", f"{int(wide['tier_total_params'].iloc[0]):,}")
        for s_, gs in wide.groupby("strategy"):
            W.num("wide" + s_.title() + "Auroc", _f(gs["fed_auroc_final"].mean()))
            W.num("wide" + s_.title() + "UploadMb", _f(gs["total_upload_mb"].mean(), 0))

    emu = [json.loads(p.read_text()) for p in sorted((R / "emulation").glob("*.json"))] if (R / "emulation").exists() else []
    if not emu:
        W.table("emulation", None); return
    # time-to-target is recomputed here from the logs: per-round analytic upload
    # time (the table shows it matches the emulated total within a few percent)
    # plus the measured compute of the slowest site, with AUROC taken by round
    # position (see netemu.time_to_target)
    from ..netemu import time_to_target
    from ..network import PROFILES
    for row in emu:
        stem = f"diabetes_mlp_wide_{row['strategy']}_R60_C20_s{row['seed']}_wide"
        sj = R / "logs" / f"{stem}_summary.json"; hc = R / "logs" / f"{stem}_history.csv"
        if not (sj.exists() and hc.exists()):
            continue
        summ = json.loads(sj.read_text()); hist = pd.read_csv(hc).to_dict("records")
        p = PROFILES[row["profile"]]
        up = [(rb["max_client_bits"] / (p.uplink_mbps * 1e6) + p.rtt_ms / 1e3) if rb["upload_bits"] > 0 else 0.0
              for rb in summ["round_bits"]]
        for c in [k for k in row if k.startswith("tt_")]:
            tgt = float(c[3:].replace("_s", ""))
            tt = time_to_target(hist, summ["round_bits"], up, tgt)
            row[c] = tt["time_to_target_s"]; row[c.replace("tt_", "rt_").replace("_s", "")] = tt["round_to_target"]
    e = pd.DataFrame(emu)
    e["relerr"] = (e["analytic_upload_s"] - e["emu_upload_s"]) / e["emu_upload_s"].where(e["emu_upload_s"] > 0)
    tcols = [c for c in e.columns if c.startswith("tt_")]
    agg = e.groupby(["profile", "strategy"]).agg(
        n=("seed", "count"), emu=("emu_upload_s", "mean"), ana=("analytic_upload_s", "mean"),
        rel=("relerr", "mean"), comp=("compute_s", "mean"), **{c: (c, "mean") for c in tcols}).reset_index()
    order = {"hospital_wan": 0, "nr_typical": 1, "nr_cell_edge": 2}
    agg = agg.sort_values(["profile", "strategy"], key=lambda c: c.map(order).fillna(9) if c.name == "profile" else c)
    agg.to_csv(T / "emulation.csv", index=False)
    pretty = {"hospital_wan": "Wired WAN", "nr_typical": "5G NR, typical", "nr_cell_edge": "5G NR, cell edge"}
    lines, prev = [], None
    for _, r in agg.iterrows():
        if prev is not None and r["profile"] != prev:
            lines.append(r"\midrule")              # one block per link profile
        prev = r["profile"]
        tt = [("never" if not np.isfinite(r[c]) else _f(r[c], 0)) for c in tcols]
        lines.append(" & ".join([pretty.get(r["profile"], r["profile"].replace("_", " ")), nm(r["strategy"]),
                                 _f(r["emu"], 1), _f(r["ana"], 1), _f(100 * r["rel"], 1), _f(r["comp"], 0)] + tt) + r" \\")
    W.table("emulation", "\n".join(lines) + "\n")
    W.num("emuMaxAbsRelErrPct", _f(100 * e["relerr"].abs().max(), 1))
    for (pf, s_), g in e.groupby(["profile", "strategy"]):
        k = "emu" + _safe(pf.replace("_", " ").title()) + s_.title()
        W.num(k + "UploadS", _f(g["emu_upload_s"].mean(), 0))
        for c in tcols:
            v = g[c].mean()
            W.num(k + "Tt" + _safe(c.replace("tt_", "").replace(".", "p").replace("_s", "")), "never" if not np.isfinite(v) else _f(v, 0))


if __name__ == "__main__":
    main()

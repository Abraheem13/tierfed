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
         "tierfed": "TierFed"}
RUN_REQUIRED = r"\textbf{[RUN REQUIRED]}"


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
                  "mean_interval", "tau_final"):
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
    return "$<10^{-4}$" if x < 1e-4 else f"{x:.3f}"


def _macro(name: str, value: str) -> str:
    name = re.sub(r"[^A-Za-z]", "", name)
    return f"\\newcommand{{\\{name}}}{{{value}}}\n"


class TexWriter:
    def __init__(self, gen_dir: Path):
        self.dir = gen_dir; self.dir.mkdir(parents=True, exist_ok=True)
        self.numbers: List[str] = []
        self.flags: Dict[str, str] = {}

    def table(self, name: str, body: Optional[str]):
        (self.dir / f"{name}.tex").write_text(body if body else RUN_REQUIRED + "\n")

    def num(self, name: str, value):
        self.numbers.append(_macro(name, value))

    def flag(self, name: str, value: bool):
        self.flags[re.sub(r"[^A-Za-z]", "", name)] = "true" if value else "false"

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
            NAMES[s], str(int(r["n_seeds"])),
            f"{_f(r.get('fed_auroc_final_mean'))} $\\pm$ {_f(r.get('fed_auroc_final_std'))}",
            _f(r.get("fed_auprc_final_mean"), 3),
            _f(r.get("fed_auroc_earlystop_value_mean")),
            _f(r.get("pooled_auroc_final_mean")),
            _f(r.get("total_upload_mb_mean"), 1),
            _f(r.get("rounds_transmitting_mean"), 0),
            "(ref.)" if s == "fedper" else _p(ni_a.get("p")),
            "(ref.)" if s == "fedper" else _p(ni_p.get("p")),
        ]) + r" \\")
    W.table(f"main_{key}", "\n".join(lines) + "\n")

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


def family(df, T, W, name, mask, keys, cols, label_fn, metrics=None):
    sub = df[mask]
    if sub.empty:
        W.table(name, None); return None
    metrics = metrics or [m for m in METRICS + NET if m in sub]
    out = summarise_runs(sub, keys, metrics)
    out.to_csv(T / f"ablation_{name}.csv", index=False)
    lines = []
    for _, r in out.iterrows():
        lines.append(" & ".join([label_fn(r)] + [c(r) for c in cols]) + r" \\")
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
           ["rho"], [ns, au, pr, up,
                     lambda r: _f(r.get("realised_payload_fraction_mean"), 2),
                     lambda r: _f(r.get("sched_K_slow_mean_mean"), 1),
                     lambda r: _f(r.get("sched_K_medium_mean_mean"), 1),
                     lambda r: _f(100 * r.get("sched_frac_rounds_tier_differentiated_mean", np.nan), 0)],
           lambda r: _f(r["rho"], 2))
    family(df, T, W, "solver", tag.str.startswith("_budget_"), ["rho", "solver"], [ns, au, up],
           lambda r: f"{_f(r['rho'], 2)} / {r['solver']}")
    family(df, T, W, "factorial", tag.str.startswith("_factorial_"), ["tag"], [ns, au, pr, up],
           lambda r: r["tag"].replace("_factorial_", "").replace("_", " "))
    family(df, T, W, "periods", tag.str.startswith("_periods_"), ["k_slow", "k_med"], [ns, au, up],
           lambda r: f"({int(r['k_slow'])}, {int(r['k_med'])})")
    family(df, T, W, "architecture", tag.str.startswith("_architecture_"), ["model", "strategy"],
           [ns, au, pr, up, lambda r: f"{int(r.get('tier_total_params_mean', 0)):,}"],
           lambda r: f"{r['model']} / {NAMES.get(r['strategy'], r['strategy'])}")
    family(df, T, W, "loss", tag.str.startswith("_loss_"), ["loss", "strategy"], [ns, au, pr, up],
           lambda r: f"{r['loss']} / {NAMES.get(r['strategy'], r['strategy'])}")
    family(df, T, W, "warmup", tag.str.startswith("_warmup_"), ["warmup"], [ns, au, pr, up],
           lambda r: str(int(r["warmup"])))
    comp = family(df, T, W, "compression", tag.str.startswith("_comp_"), ["strategy", "compressor"],
                  [ns, au, pr, up], lambda r: f"{NAMES.get(r['strategy'], r['strategy'])} / {r['compressor']}")
    family(df, T, W, "dp", tag.str.startswith("_dp"), ["strategy", "dp_sigma"],
           [ns, lambda r: _f(r.get("dp_epsilon_mean"), 1), lambda r: _f(r.get("dp_rounds_charged_mean"), 0), au, pr],
           lambda r: f"{NAMES.get(r['strategy'], r['strategy'])} / {_f(r['dp_sigma'], 1)}",
           metrics=[m for m in METRICS if m in df] + ["dp_epsilon", "dp_rounds_charged"])
    family(df, T, W, "attack", tag == "_attack", ["strategy"],
           [ns, lambda r: f"{int(r.get('n_params_exposed_mean_mean', 0)):,}",
            lambda r: _f(r.get("inversion_nmse_mean_mean"), 2),
            lambda r: _f(r.get("inversion_feature_corr_mean_mean"), 3),
            lambda r: _f(r.get("mia_auc_mean_mean"), 3), up],
           lambda r: NAMES.get(r["strategy"], r["strategy"]))
    family(df, T, W, "scaling", tag.str.startswith("_scale"), ["n_clients", "strategy"],
           [ns, au, up, lambda r: _f(r.get("s_per_round_mean"), 2), lambda r: _f(r.get("peak_mem_mb_mean"), 0)],
           lambda r: f"{int(r['n_clients'])} / {NAMES.get(r['strategy'], r['strategy'])}")

    W.finish()
    print(f"collected {len(df)} runs -> {T} and {a.paper}")
    missing = [p.name for p in Path(a.paper).glob("*.tex") if "RUN REQUIRED" in p.read_text()]
    print("tables still waiting for runs:", missing if missing else "none")


if __name__ == "__main__":
    main()

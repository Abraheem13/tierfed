"""Build summary tables and statistical tests from results ALREADY on disk."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .stats import compare_all, non_inferiority, summarise_runs

METRICS = ["fed_auroc_final", "fed_auroc_best", "fed_auroc_ppd",
           "fed_auroc_earlystop_value", "fed_auprc_final", "fed_auprc_best",
           "fed_f1_final", "fed_f1_best_final", "fed_ece_final",
           "global_auroc_final", "total_upload_mb", "divergence_total_mean"]


def load_summaries(log_dir):
    rows: List[Dict] = []
    for p in sorted(Path(log_dir).glob("*_summary.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        cfg = d.get("config", {})
        row = {"file": p.name, "dataset": cfg.get("dataset"), "model": cfg.get("model"),
               "strategy": cfg.get("strategy"), "seed": cfg.get("seed"),
               "rounds": cfg.get("rounds"), "n_clients": cfg.get("n_clients"),
               "loss": cfg.get("loss"), "compressor": cfg.get("compressor") or "none",
               "tag": cfg.get("tag", "")}
        sk = cfg.get("strategy_kwargs") or {}
        row["k_slow"] = sk.get("k_slow"); row["k_med"] = sk.get("k_med")
        row["private_head"] = sk.get("private_head"); row["schedule"] = sk.get("schedule")
        row["dp_sigma"] = (d.get("dp") or {}).get("dp_sigma", 0.0)
        row["dp_epsilon"] = (d.get("dp") or {}).get("dp_epsilon")
        row.update(d.get("summary", {}))
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Aggregate finished runs into tables")
    ap.add_argument("--results", default="results")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--baselines", nargs="+", default=["fedavg", "fedprox", "fedper", "fedlama"])
    a = ap.parse_args(argv)

    R = Path(a.results)
    df = load_summaries(R / "logs")
    if df.empty:
        print("no *_summary.json found — nothing to collect"); return
    if a.dataset: df = df[df.dataset == a.dataset]
    if a.model: df = df[df.model == a.model]

    T = R / "tables"; T.mkdir(parents=True, exist_ok=True)
    tag = f"{a.dataset or 'all'}_{a.model or 'all'}"
    df.to_csv(T / f"collected_{tag}_raw.csv", index=False)
    metrics = [m for m in METRICS if m in df]

    main_df = df[df["tag"].fillna("") == ""]
    if not main_df.empty:
        summarise_runs(main_df, ["strategy"], metrics).to_csv(
            T / f"benchmark_{tag}_summary.csv", index=False)
        tests = []
        for base in a.baselines:
            if base in main_df.strategy.values:
                tests.append(compare_all(main_df, "strategy", base, metrics))
        if tests:
            allt = pd.concat(tests, ignore_index=True)
            allt.to_csv(T / f"benchmark_{tag}_tests.csv", index=False)
            key = allt[allt.metric == "fed_auroc_final"]
            print("\n=== NFL vs baselines (final AUROC, Holm-corrected) ===")
            cols = [c for c in ["comparison", "mean_diff", "diff_ci_lo", "diff_ci_hi",
                                "t_p", "p_holm", "significant_holm", "hedges_g"] if c in key]
            print(key[cols].to_string(index=False))
        ni = {}
        for base in a.baselines:
            if base in main_df.strategy.values:
                x = main_df[main_df.strategy == "nested"].sort_values("seed")
                y = main_df[main_df.strategy == base].sort_values("seed")
                n = min(len(x), len(y))
                if n >= 2:
                    ni[f"nested_vs_{base}"] = non_inferiority(
                        x["fed_auroc_best"].values[:n], y["fed_auroc_best"].values[:n], 0.01)
        (T / f"benchmark_{tag}_noninferiority.json").write_text(json.dumps(ni, indent=2))

    fam = {"factorial": df[df["tag"].fillna("").str.contains("_fac_")],
           "periods": df[df["tag"].fillna("").str.contains("_ks")],
           "loss": df[df["tag"].fillna("").str.contains("_loss_")],
           "compression": df[df["tag"].fillna("").str.contains("_comp_")],
           "privacy": df[df["tag"].fillna("").str.contains("_dp")],
           "tierfed": df[df["tag"].fillna("").str.contains("_tf_")]}
    for name, sub in fam.items():
        if sub.empty: continue
        keys = {"factorial": ["tag"], "periods": ["k_slow", "k_med"],
                "loss": ["loss", "strategy"], "compression": ["strategy", "compressor"],
                "privacy": ["strategy", "dp_sigma"],
                "tierfed": ["strategy", "tag"]}[name]
        keys = [k for k in keys if k in sub and sub[k].notna().any()] or ["tag"]
        out = summarise_runs(sub, keys, metrics)
        out.to_csv(T / f"ablation_{name}_summary.csv", index=False)
        print(f"\n=== {name} ({len(sub)} runs) ===")
        show = [c for c in out.columns if c.endswith("_mean") or c in keys][:7]
        print(out[show].to_string(index=False))

    print(f"\ncollected {len(df)} runs -> {T}")


if __name__ == "__main__":
    main()

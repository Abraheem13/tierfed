"""Paired DP-vs-local comparison on the DP seeds (42-47), read from the run logs.

Prints, per method, the mean final-round and validation-selected AUROC over the
six DP seeds, and local training's values on the same seeds, so the private
comparison in the TMLCN manuscript is paired. Reads only results/logs.
"""
import glob, json, sys
from statistics import mean

LOG = sys.argv[1] if len(sys.argv) > 1 else "results/logs"
SEEDS = range(42, 48)
ARMS = {"FedPer": "fedper_R60_C20_s{s}_au_dpe4_q1_fedper",
        "Fixed (4,4)": "fixed_R60_C20_s{s}_au_dpe4_q1_fixed44",
        "Fixed (8,8)": "fixed_R60_C20_s{s}_au_dpe4_q1_fixed88",
        "One-shot": "fixed_R60_C20_s{s}_au_dpe4_q1_oneshot",
        "TierFed-DP": "tierfed_dp_R60_C20_s{s}_au_dpe4_q1_tierfed_dp",
        "Local only": "local_R60_C20_s{s}_au_local"}

def load(stem, s):
    p = f"{LOG}/diabetes_mlp_{stem.format(s=s)}_summary.json"
    return json.load(open(p))["summary"]

local = {s: load(ARMS["Local only"], s) for s in SEEDS}
for name, stem in ARMS.items():
    fin, sel, rnd, gap = [], [], [], []
    for s in SEEDS:
        d = load(stem, s)
        fin.append(d["fed_auroc_final"]); sel.append(d["fed_auroc_valsel"])
        rnd.append(d.get("fed_val_auroc_valsel_round", d.get("valsel_round", float("nan"))))
        gap.append(d["fed_auroc_final"] - local[s]["fed_auroc_final"])
    print(f"{name:12s} final={mean(fin):.4f} valsel={mean(sel):.4f} "
          f"final-local(paired)={mean(gap):+.4f} min={min(gap):+.4f} max={max(gap):+.4f} n={len(fin)}")

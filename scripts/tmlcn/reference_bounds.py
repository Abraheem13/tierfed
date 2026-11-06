"""Centralised MLP and gradient boosting on the frontier's shared seeds (42-46) and on all ten.

The frontier of Fig. 5(a) is drawn on the five seeds that every frontier configuration
shares; its grey reference lines use the same five seeds. Reads only results/logs.
"""
import glob, json, sys
from statistics import mean
LOG = sys.argv[1] if len(sys.argv) > 1 else "results/logs"
for name in ("central", "hgb"):
    v = {}
    for p in glob.glob(f"{LOG}/diabetes_mlp_{name}_R60_C20_s*_au_bound_summary.json"):
        s = int(p.split("_R60_C20_s")[1].split("_")[0])
        v[s] = json.load(open(p))["summary"]["fed_auroc_valsel"]
    five = [v[s] for s in range(42, 47) if s in v]
    print(f"{name:8s} seeds42-46 n={len(five)} mean={mean(five):.4f}   all n={len(v)} mean={mean(v.values()):.4f}")

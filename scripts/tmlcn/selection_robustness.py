"""Robustness of the validation-selected reading to the selection rule.

The main protocol selects r_hat = first argmax_r A^V(r) (Eq. (5)). On a flat
validation curve a tiny numerical difference can move r_hat, which is how one
run moved by 0.0893 AUROC across machines. This script re-reads every run of
the main comparison and of the ResNet family from its per-round history and
compares three rules:

  argmax   r_hat = first argmax of A^V(r)                 (the paper's rule)
  smooth3  r_hat = first argmax of the trailing 3-round mean of A^V
  last10   mean of A^T over the last ten rounds            (no selection)

It prints, per method, the mean A^T at each rule, the paired difference to
FedPer and to Fixed (4,4), and checks that the argmax rule reproduces the
logged summary value (fed_auroc_valsel). Reads only results/logs.
"""
import csv
import glob
import json
import sys
from statistics import mean

LOG = sys.argv[1] if len(sys.argv) > 1 else "results/logs"


def rows(path):
    with open(path) as f:
        r = list(csv.DictReader(f))
    va = [float(x["fed_val_auroc"]) if x["fed_val_auroc"] not in ("", "nan") else float("-inf") for x in r]
    te = [float(x["fed_auroc"]) if x["fed_auroc"] not in ("", "nan") else float("nan") for x in r]
    return va, te


def argmax_first(v):
    best = max(v)
    return v.index(best)


def readings(path):
    va, te = rows(path)
    i = argmax_first(va)
    sm = [mean(va[max(0, j - 2): j + 1]) for j in range(len(va))]
    k = argmax_first(sm)
    return {"argmax": te[i], "smooth3": te[k], "last10": mean(te[-10:]),
            "r_argmax": i + 1, "r_smooth3": k + 1}


def family(model, sites, stems, seeds, tag="au"):
    out = {}
    for name, (strat, stem) in stems.items():
        per = {}
        for s in seeds:
            hp = f"{LOG}/diabetes_{model}_{strat}_R60_C{sites}_s{s}_{tag}_{stem}_history.csv"
            sp = hp.replace("_history.csv", "_summary.json")
            if not glob.glob(hp):
                continue
            r = readings(hp)
            r["logged"] = json.load(open(sp))["summary"].get("fed_auroc_valsel", float("nan"))
            per[s] = r
        out[name] = per
    return out


def report(title, fam, refs=("FedPer", "Fixed (4,4)")):
    print(f"\n== {title} ==")
    print(f"{'method':28s} n  argmax  smooth3 last10 | check(max|argmax-logged|) | d_smooth3 vs " + ", ".join(refs))
    for name, per in fam.items():
        if not per:
            continue
        seeds = sorted(per)
        a = mean(per[s]["argmax"] for s in seeds)
        b = mean(per[s]["smooth3"] for s in seeds)
        c = mean(per[s]["last10"] for s in seeds)
        chk = max(abs(per[s]["argmax"] - per[s]["logged"]) for s in seeds)
        ds = []
        for ref in refs:
            rp = fam.get(ref, {})
            common = [s for s in seeds if s in rp]
            ds.append(f"{mean(per[s]['smooth3'] - rp[s]['smooth3'] for s in common):+.4f}" if common else "  --  ")
        print(f"{name:28s} {len(seeds):2d} {a:.4f} {b:.4f} {c:.4f} | {chk:.1e} | " + " ".join(ds))


MAIN = {"Local only": ("local", "local"), "FedAvg": ("fedavg", "fedavg"), "FedProx": ("fedprox", "fedprox"),
        "FedPer": ("fedper", "fedper"), "FedLAMA-style": ("fedlama", "fedlama"),
        "AdaComm": ("adacomm", "adacomm8"), "LAG, equal weights": ("lag", "lag_uniform"), "LAG": ("lag", "lag"),
        "Fixed (4,4)": ("fixed", "fixed44"), "Fixed (4,4), staggered": ("fixed", "fixed44_st"),
        "Fixed (8,8)": ("fixed", "fixed88"), "TierFed": ("tierfed", "tierfed"),
        "TierFed, staggered": ("tierfed", "tierfed_st")}

if __name__ == "__main__":
    report("Main comparison, MLP, specialty sites", family("mlp", 20, MAIN, range(42, 52)))


RESNET = {"Local only": ("local", "bb_local"), "FedPer": ("fedper", "bb_fedper"),
          "Fixed (4,4)": ("fixed", "bb_fixed44"), "Fixed (1,5)": ("fixed", "bb_fixed15"),
          "TierFed": ("tierfed", "bb_tierfed")}


def hetero(alpha, sites):
    return {"Local only": ("local", f"het_a{alpha}_local"), "FedPer": ("fedper", f"het_a{alpha}_fedper"),
            "Fixed (4,4)": ("fixed", f"het_a{alpha}_fixed44"), "Fixed (8,8)": ("fixed", f"het_a{alpha}_fixed88"),
            "TierFed": ("tierfed", f"het_a{alpha}_tierfed")}


if __name__ == "__main__":
    report("ResNet backbone, specialty sites", family("resnet", 20, RESNET, range(42, 52)),
           refs=("FedPer", "Fixed (4,4)"))
    for a in ("0.3", "1", "10"):
        report(f"Dirichlet alpha={a}", family("mlp", 20, hetero(a, 20), range(42, 52)),
               refs=("Fixed (4,4)", "Fixed (8,8)"))

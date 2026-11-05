"""Audit batch (v6).

Every run here uses a per-site, patient-grouped train/validation/test split
(70/10/20 % of each site's patients). The round reported for a run is chosen on
validation; test data never choose anything. All tags start with "_au_", so the
earlier logs stay untouched and reproducible.

Stages
  main      the main comparison, with the corrected fixed period (one due rule,
            warm-up 5), the staggered period, the TierFed variants, local-only,
            LAG with sample weighting, and final-round fine-tuning controls (E-1..E-4)
  frontier  every method with its communication knob swept, for the iso-byte
            frontier and an equal per-method validation tuning budget (E-5, E-6)
  hetero    Dirichlet label skew alpha in {0.1, 0.3, 1, 10} (decision gate, B-1)
  dp        matched-epsilon DP with q = 1 as the primary cross-silo setting, the
            sparse-period sweep up to one-shot federation, and a clipping
            sensitivity (D-3, D-4, D-5)
  derma     DermaMNIST, malignant {akiec, bcc, mel} against benign, alpha = 0.1 (B-3)
  backbone  ResNet and FT-Transformer backbones (run on the GPU), where the tiers'
            payload and dispersion shares differ (M-2, T-9)
  timing    timing side channel: a shifted target site against the schedule it induces (D-8)
  peak      staggered tiers with an 8-bit warm-up, so no round sends the full payload (M-1, M-6),
            and the byte quota counted over all rounds, warm-up included (P0-5)

One process runs one stage for one seed (and one alpha in the hetero stage);
scripts/run_audit.ps1 runs them in parallel and skips finished runs.
"""
from __future__ import annotations

import argparse
import itertools

from ..utils import get_logger
from .runner import run_single

log = get_logger()

TW = 5
VAL = dict(val_split=True, client_test_frac=0.3, val_share=1 / 3)
TF = {"warmup": TW, "rho": 0.25}
FT = [1, 3]                      # E-3: local fine-tuning epochs evaluated at the final round
ALPHAS = (0.1, 0.3, 1.0, 10.0)


def al(ks: int, km: int, **kw):
    """Fixed period on the shared due rule: warm-up T_w, then (r - T_w - phi) mod K = 0."""
    return {"k_slow": ks, "k_med": km, "warmup": TW, "aligned": True, **kw}


def main_configs():
    return [
        ("fedavg", {}, "fedavg"),
        ("fedprox", {"mu": 0.01}, "fedprox"),
        ("scaffold", {}, "scaffold"),
        ("fedper", {}, "fedper"),
        ("fedlama", {"tau": 5, "phi": 2}, "fedlama"),
        ("adacomm", {"tau0": 8, "max_interval": 8}, "adacomm8"),
        ("lag", {"xi": 1.0}, "lag_uniform"),                 # as in the earlier runs (equal weights)
        ("lag", {"xi": 1.0, "weighted": True}, "lag"),
        ("fixed", {"k_slow": 5, "k_med": 2}, "fixed52_legacy"),   # the earlier rule, for continuity
        ("fixed", al(5, 2), "fixed52"),
        ("fixed", al(4, 4), "fixed44"),
        ("fixed", al(4, 4, stagger=True), "fixed44_st"),
        ("fixed", al(8, 8), "fixed88"),
        ("tierfed", TF, "tierfed"),
        ("tierfed", {**TF, "stagger": True}, "tierfed_st"),
        ("tierfed", {**TF, "debias": True}, "tierfed_db"),
        ("tierfed", {**TF, "peak_cap": 0.6}, "tierfed_pk"),
        ("local", {}, "local"),
    ]


def frontier_configs():
    done = {(4, 4), (8, 8), (1, 1)}
    cfgs = [("fixed", al(ks, km), f"fixed{ks}{km}")
            for ks, km in itertools.product((1, 2, 4, 8), repeat=2) if (ks, km) not in done]
    cfgs += [("fixed", al(k, k), f"fixed{k}{k}") for k in (3, 6)]
    cfgs += [("adacomm", {"tau0": t, "max_interval": max(t, 8)}, f"adacomm{t}") for t in (2, 4, 16)]
    cfgs += [("lag", {"xi": x, "weighted": True}, f"lag_xi{x}") for x in (10, 100, 1000)]
    cfgs += [("fedlama", {"tau": t, "phi": 2}, f"fedlama_tau{t}") for t in (2, 10)]
    cfgs += [("tierfed", {**TF, "rho": r}, f"tierfed_rho{r}") for r in (0.15, 0.4, 0.6)]
    cfgs += [("fedprox", {"mu": m}, f"fedprox_mu{m}") for m in (0.001, 0.1, 1.0)]
    cfgs += [("fedper", {}, "fedper_q8", {"compressor": "quant8"}),
             ("tierfed", TF, "tierfed_q8", {"compressor": "quant8"})]
    return cfgs


def hetero_configs():
    return [("fedper", {}, "fedper"), ("fedavg", {}, "fedavg"), ("local", {}, "local"),
            ("fixed", al(2, 2), "fixed22"), ("fixed", al(4, 4), "fixed44"), ("fixed", al(8, 8), "fixed88"),
            ("tierfed", {**TF, "rho": 0.15}, "tierfed_rho0.15"), ("tierfed", TF, "tierfed"),
            ("tierfed", {**TF, "rho": 0.4}, "tierfed_rho0.4"),
            ("tierfed", {**TF, "debias": True}, "tierfed_db"), ("tierfed", {**TF, "stagger": True}, "tierfed_st"),
            ("adacomm", {"tau0": 4, "max_interval": 8}, "adacomm4"),
            ("adacomm", {"tau0": 8, "max_interval": 8}, "adacomm8")]


def dp_configs(rounds: int):
    """(strategy, kwargs, name, q, epsilon, clip). q = 1 is primary (D-3)."""
    out = []
    def planner(eps, q):
        return dict(rounds=rounds, warmup=TW, epsilon=eps, q=q, lam=1.0, rho=1.0)
    for eps in (1.0, 2.0, 4.0, 8.0):
        out += [("fedavg", {}, "fedavg", 1.0, eps, 1.0), ("fedper", {}, "fedper", 1.0, eps, 1.0),
                ("fixed", al(4, 4), "fixed44", 1.0, eps, 1.0), ("fixed", al(8, 8), "fixed88", 1.0, eps, 1.0),
                ("tierfed_dp", planner(eps, 1.0), "tierfed_dp", 1.0, eps, 1.0)]
        # D-5: sparser periods, and one-shot federation (no warm-up, one release at R)
        if eps in (1.0, 4.0, 8.0):
            out += [("fixed", al(k, k), f"fixed{k}{k}", 1.0, eps, 1.0) for k in (12, 20, 30)]
            out += [("fixed", {"k_slow": rounds, "k_med": rounds, "warmup": 0, "aligned": True},
                     "oneshot", 1.0, eps, 1.0)]
    # clipping sensitivity at epsilon = 4 (C = 1 is in the grid above)
    for c in (0.3, 3.0):
        out += [("fedper", {}, f"fedper_c{c}", 1.0, 4.0, c), ("fixed", al(8, 8), f"fixed88_c{c}", 1.0, 4.0, c),
                ("tierfed_dp", planner(4.0, 1.0), f"tierfed_dp_c{c}", 1.0, 4.0, c)]
    return out


def frontier2_configs():
    """Second-night additions to the frontier: LAG thresholds at which 20 sites
    actually skip uploads (xi <= 100 never skips, xi = 1000 rarely does)."""
    return [("lag", {"xi": x, "weighted": True}, f"lag_xi{x}") for x in (2000, 5000, 10000)]


def dropout_configs():
    """N-7: sites drop out of a round at random (uniform participation p per round)."""
    out = []
    for p in (0.5, 0.8):
        out += [(s, k, f"drop{p:g}_{n}", {"participation": p})
                for s, k, n in (("fedper", {}, "fedper"), ("fixed", al(4, 4), "fixed44"), ("tierfed", TF, "tierfed"),
                                ("adacomm", {"tau0": 8, "max_interval": 8}, "adacomm8"))]
    return out


def peak_configs():
    """M-1 with M-6: staggered tiers bound the post-warm-up peak; an 8-bit warm-up
    bounds the warm-up peak, so the peak over ALL rounds falls below the full payload."""
    w8 = {"warmup_compressor": "quant8"}
    return [("tierfed", {**TF, **w8}, "tierfed_w8"),
            ("tierfed", {**TF, "stagger": True, **w8}, "tierfed_st_w8"),
            ("tierfed", {**TF, "peak_cap": 0.6, **w8}, "tierfed_pk_w8"),
            ("fixed", al(4, 4, **w8), "fixed44_w8"),
            ("fixed", al(4, 4, stagger=True, **w8), "fixed44_st_w8"),
            # P0-5: the quota over all R rounds, warm-up included
            ("tierfed", {**TF, "quota": "all", "rounds": 60}, "tierfed_all"),
            ("tierfed", {**TF, "quota": "all", "rounds": 60, "rho": 0.4}, "tierfed_all_rho0.4"),
            ("tierfed", {**TF, "quota": "all", "rounds": 60, "warmup_payload": 0.25, **w8}, "tierfed_all_w8"),
            ("tierfed", {**TF, "quota": "all", "rounds": 60, "debias": True}, "tierfed_all_db")]


FROZEN = {"resnet": (1, 5), "transformer": (2, 6)}   # TierFed's modal choice in the v4 backbone runs


def backbone_configs(model: str):
    """Deeper backbones, where the payload and dispersion shares of the tiers differ
    (the regime in which Proposition 2 predicts a gain from differentiation). The
    fixed period at TierFed's own modal intervals separates the choice of intervals
    from their adaptation during training."""
    ks, km = FROZEN[model]
    return ([("fedper", {}, "fedper"), ("fedavg", {}, "fedavg"), ("local", {}, "local")]
            + [("fixed", al(k, k), f"fixed{k}{k}") for k in (2, 3, 4, 6, 8)]
            + [("fixed", al(ks, km), f"fixed{ks}{km}"),
               ("tierfed", TF, "tierfed"), ("tierfed", {**TF, "rho": 0.15}, "tierfed_rho0.15"),
               ("tierfed", {**TF, "rho": 0.4}, "tierfed_rho0.4"), ("tierfed", {**TF, "debias": True}, "tierfed_db")])


def timing_configs():
    """D-8: can a passive observer of the traffic infer a site's data from the
    schedule? Each method runs with and without a shifted target site; fixed44 is
    the data-independent control (its schedule cannot differ between the worlds)."""
    return [("tierfed", TF, "tierfed"), ("adacomm", {"tau0": 8, "max_interval": 8}, "adacomm8"),
            ("lag", {"xi": 2000, "weighted": True}, "lag_xi2000"), ("fixed", al(4, 4), "fixed44")]


def derma_configs():
    return [("fedavg", {}, "fedavg"), ("fedper", {}, "fedper"), ("local", {}, "local"),
            ("fixed", al(4, 4), "fixed44"), ("tierfed", TF, "tierfed")]


def _run(base, strat, skw, name, extra=None):
    cfg = {**base, **VAL, "strategy": strat, "strategy_kwargs": skw, "tag": f"_au_{name}", **(extra or {})}
    return run_single(cfg, skip_existing=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Audit batch (v6)")
    ap.add_argument("--stage", required=True,
                    choices=["main", "frontier", "hetero", "dp", "derma", "peak", "backbone", "timing",
                             "frontier2", "dropout"])
    ap.add_argument("--model", default="mlp", choices=["mlp", "resnet", "transformer"],
                    help="backbone stage: resnet or transformer; timing stage: mlp or resnet")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 52)))
    ap.add_argument("--alpha", type=float, default=None, help="hetero stage: one Dirichlet alpha")
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--clients", type=int, default=20)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results")
    ap.add_argument("--only", nargs="+", default=None,
                    help="keep only these configurations (base names, e.g. fedper tierfed fixed44)")
    ap.add_argument("--eps", nargs="+", type=float, default=None, help="dp stage: keep only these epsilons")
    ap.add_argument("--q", nargs="+", type=float, default=None, help="dp stage: keep only these sampling rates")
    ap.add_argument("--count", action="store_true", help="print the number of runs and exit")
    ap.add_argument("--smoke", action="store_true", help="tiny synthetic runs to test the pipeline")
    a = ap.parse_args(argv)

    base = dict(dataset="diabetes", model="mlp", rounds=a.rounds, n_clients=a.clients,
                device=a.device, out_dir=a.out)
    syn = dict(n_per_client=120, n_features=20)
    if a.smoke:
        base.update(dataset="synthetic", dataset_kwargs=syn, rounds=8, n_clients=4)
    if a.stage == "main":
        jobs = [(s, k, n, {"eval_finetune_epochs": FT}) for s, k, n in main_configs()]
    elif a.stage == "backbone":
        if not a.smoke:
            base["model"] = a.model
        jobs = [(s, k, f"bb_{n}", None) for s, k, n in backbone_configs(a.model)]
    elif a.stage == "timing":
        if a.model != "resnet" or a.smoke:
            confs = timing_configs()
        else:                                   # where TierFed differentiates (GPU)
            base["model"] = "resnet"; confs = [("tierfed", TF, "tierfed")]
        jobs = [(s, k, f"tm_{w}_{n}", {"perturb_site": {"site": 0, "flip": 0.5}} if w == "shift" else None)
                for w in ("clean", "shift") for s, k, n in confs]
    elif a.stage == "dropout":
        jobs = [(s, k, n, extra) for s, k, n, extra in dropout_configs()]
    elif a.stage == "frontier2":
        jobs = [(s, k, n, None) for s, k, n in frontier2_configs()]
    elif a.stage == "peak":
        jobs = [(s, k, n, None) for s, k, n in peak_configs()]
    elif a.stage == "frontier":
        jobs = [(c[0], c[1], c[2], c[3] if len(c) > 3 else None) for c in frontier_configs()]
    elif a.stage == "hetero":
        alphas = [a.alpha] if a.alpha is not None else list(ALPHAS)
        jobs = [(s, k, f"het_a{al_:g}_{n}", {"dataset_kwargs": {**(syn if a.smoke else {}),
                                                                 "partition": "dirichlet", "alpha": al_}})
                for al_ in alphas for s, k, n in hetero_configs()]
    elif a.stage == "dp":
        jobs = []
        for s, k, n, q, eps, clip in dp_configs(base["rounds"]):
            if (a.eps and eps not in a.eps) or (a.q and q not in a.q) or (a.only and n not in a.only):
                continue
            dp = dict(enabled=True, sigma=1.0, max_norm=clip, delta=1e-5, sample_rate=q, target_epsilon=eps)
            extra = {"dp": dp}
            if n == "oneshot":
                extra["common_init"] = True
            jobs.append((s, k, f"dpe{eps:g}_q{q:g}_{n}", extra))
    else:
        dkw = {"flag": "dermamnist", "positive_classes": [0, 1, 4], "partition": "dirichlet", "alpha": 0.1}
        if a.smoke:
            dkw = syn
        else:
            base.update(dataset="medmnist", model="cnn")
        jobs = [(s, k, f"derma_a0.1_{n}", {"dataset_kwargs": dkw}) for s, k, n in derma_configs()]

    if a.only and a.stage != "dp":
        def base_name(full):
            if full.startswith("drop"):                          # drop0.5_<name>
                return full.split("_", 1)[1]
            for pre in ("het_a", "derma_a"):                      # het_a0.1_<name>
                if full.startswith(pre):
                    return full.split("_", 2)[2] if full.count("_") >= 2 else full
            for pre in ("bb_", "tm_clean_", "tm_shift_"):
                if full.startswith(pre):
                    return full[len(pre):]
            return full
        jobs = [j for j in jobs if base_name(j[2]) in a.only]
    if a.count:
        print(len(jobs) * len(a.seeds))
        return
    for seed in a.seeds:
        for strat, skw, name, extra in jobs:
            _run({**base, "seed": seed}, strat, skw, name, extra)


if __name__ == "__main__":
    main()

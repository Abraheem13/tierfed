"""Tests for the audit fixes: one due rule for every planned schedule, the
staggered schedule, the local-only bound, the three-way patient-grouped split
and the validation-selected, phase-matched, calibrated and per-site metrics."""
import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")

from src.experiments.runner import run_single
from src.strategies.adaptive import PrivacyPlannedTierFed
from src.strategies.nested_fl import FixedSchedule
from src.strategies.schedule_rules import due_tiers, peak_fraction, release_rounds, stagger_phases

BASE = dict(dataset="synthetic", dataset_kwargs=dict(n_per_client=80, n_features=20),
            model="mlp", rounds=6, n_clients=4, seed=42, out_dir="/tmp/tierfed_test_audit", device="cpu")

TIERS = {"a.w": "slow", "b.w": "medium", "c.w": "fast"}
SIZES = {"a.w": 50, "b.w": 30, "c.w": 20}


@pytest.mark.parametrize("K,count", [(4, 18), (8, 11), (3, 23)])
def test_release_count_of_the_shared_rule(K, count):
    # T_w = 5 full rounds, then every K-th round: 5 + floor((60 - 5) / K)
    assert len(release_rounds({"slow": K, "medium": K}, 60, 5)) == count


def test_aligned_fixed_period_realises_its_plan():
    fx = FixedSchedule(list(TIERS), SIZES, TIERS, k_slow=4, k_med=4, warmup=5, aligned=True)
    sent = [r for r in range(1, 61) if fx.send_keys(r)]
    assert sent == release_rounds({"slow": 4, "medium": 4}, 60, 5)
    assert len(sent) == 18


def test_legacy_fixed_rule_is_unchanged():
    # v3-v12 logs: round 1, then r mod K == 0, no warm-up -> 1 + 12 rounds for K = 5
    fx = FixedSchedule(list(TIERS), SIZES, TIERS, k_slow=5, k_med=5)
    assert len([r for r in range(1, 61) if fx.send_keys(r)]) == 13


def test_planner_and_aligned_fixed_share_one_rule():
    p = PrivacyPlannedTierFed(list(TIERS), SIZES, TIERS, rounds=60, warmup=5, epsilon=4.0, k_max=8)
    for r in range(1, 61):
        assert p.due(r) == due_tiers(r, p.K, 5)


def test_stagger_separates_tiers_with_a_common_interval():
    K = {"slow": 4, "medium": 4}
    w = {"slow": 0.6, "medium": 0.4}
    ph = stagger_phases(K)
    assert ph["slow"] != ph["medium"]
    assert peak_fraction(K, w, 60, 5) == pytest.approx(1.0)
    assert peak_fraction(K, w, 60, 5, ph) == pytest.approx(0.6)
    # co-prime intervals always coincide somewhere: no phase removes the overlap
    assert stagger_phases({"slow": 3, "medium": 2}) == {"medium": 0, "slow": 0}


def test_three_way_split_is_disjoint_and_patient_grouped():
    from src.data import load_dataset
    ds = load_dataset("synthetic", n_clients=4, seed=1, n_per_client=80, n_features=20)
    ds.groups = np.repeat(np.arange(len(ds.y_train) // 2 + 1), 2)[: len(ds.y_train)]
    tr, va, te = ds.client_val_test_split(frac=0.3, val_share=1 / 3, seed=3)
    for k, idx in enumerate(ds.client_indices):
        a, b, c = set(tr[k].tolist()), set(va[k].tolist()), set(te[k].tolist())
        assert not (a & b) and not (a & c) and not (b & c)
        assert a | b | c == set(idx.tolist())
        ga, gb, gc = (set(ds.groups[list(s)].tolist()) for s in (a, b, c))
        assert not (ga & gb) and not (ga & gc) and not (gb & gc)
        assert len(b) > 0 and len(c) > 0


def test_local_only_sends_nothing():
    s = run_single({**BASE, "strategy": "local"}, robust=False)["summary"]
    assert s["total_upload_mb"] == 0 and s["rounds_transmitting"] == 0
    assert np.isfinite(s["fed_auroc_final"]) and np.isfinite(s["fed_auroc_phase"])


def test_validation_selected_and_calibrated_metrics():
    cfg = {**BASE, "strategy": "fedper", "val_split": True, "client_test_frac": 0.3, "val_share": 1 / 3}
    res = run_single(cfg, robust=False)
    s = res["summary"]
    for k in ("fed_auroc_valsel", "valsel_round", "upload_mb_valsel", "fed_auroc_phase",
              "fed_auroc_last20_mean", "fed_cal_ece_final", "fed_cal_slope_final",
              "fed_auroc_site_min_final", "fed_auroc_site_p10_final"):
        assert k in s, k
    r = int(s["valsel_round"])
    h = res["history"][r - 1]
    assert s["fed_auroc_valsel"] == h["fed_auroc"]
    assert s["upload_mb_valsel"] == h["upload_mb"] <= s["total_upload_mb"]
    # the selected round maximises VALIDATION AUROC (earliest on ties)
    v = [x["fed_val_auroc"] for x in res["history"]]
    assert h["fed_val_auroc"] == max(v) and v.index(max(v)) == r - 1


def test_finetune_control_leaves_the_trajectory_unchanged():
    a = run_single({**BASE, "strategy": "fedavg"}, robust=False)["summary"]
    b = run_single({**BASE, "strategy": "fedavg", "eval_finetune_epochs": [1, 3], "tag": "_ft13"},
                   robust=False)["summary"]
    assert a["fed_auroc_final"] == b["fed_auroc_final"]
    for j in (1, 3):
        assert np.isfinite(b[f"fed_ft{j}_auroc_final"])


def test_lag_weighting_option():
    for w in (False, True):
        s = run_single({**BASE, "strategy": "lag", "tag": f"_w{int(w)}",
                        "strategy_kwargs": {"weighted": w}}, robust=False)
        assert s["schedule"]["weighted"] == float(w)


def test_round_time_peak_metrics():
    from src.network import cost_run
    rb = [{"max_client_bits": 8e6, "upload_bits": 8e7, "n_senders": 10},
          {"max_client_bits": 0.0, "upload_bits": 0.0, "n_senders": 0},
          {"max_client_bits": 4e6, "upload_bits": 4e7, "n_senders": 10}]
    c = cost_run(rb, "nr_cell_edge")
    assert c["net_nr_cell_edge_round_time_max_s"] == pytest.approx(8e6 / 5e6 + 0.03)
    assert c["net_nr_cell_edge_round_time_p95_s"] == pytest.approx(8e6 / 5e6 + 0.03)


@pytest.mark.parametrize("kw", [{"stagger": True}, {"debias": True}, {"peak_cap": 0.8}])
def test_tierfed_variants_run(kw):
    s = run_single({**BASE, "strategy": "tierfed", "tag": "_" + "_".join(kw),
                    "strategy_kwargs": {"warmup": 1, "rho": 0.3, **kw}}, robust=False)["summary"]
    assert s["total_upload_mb"] > 0 and np.isfinite(s["fed_auroc_final"])


def test_warmup_compressor_cuts_only_warmup_bytes():
    kw = {"warmup": 3, "rho": 0.3}
    a = run_single({**BASE, "strategy": "tierfed", "tag": "_wc0", "strategy_kwargs": kw}, robust=False)
    b = run_single({**BASE, "strategy": "tierfed", "tag": "_wc8",
                    "strategy_kwargs": {**kw, "warmup_compressor": "quant8"}}, robust=False)
    ra, rb = a["round_bits"], b["round_bits"]
    for r in range(3):              # warm-up rounds: 8-bit, about a quarter of the float32 bits
        assert rb[r]["upload_bits"] < 0.3 * ra[r]["upload_bits"]
    assert b["summary"]["total_upload_mb"] < a["summary"]["total_upload_mb"]


@pytest.mark.parametrize("strategy,kw", [("fixed", {"k_slow": 2, "k_med": 3, "warmup": 2, "aligned": True}),
                                         ("fixed", {"k_slow": 4, "k_med": 4, "warmup": 2, "stagger": True}),
                                         ("fixed", {"k_slow": 3, "k_med": 2}),
                                         ("fedper", {}), ("local", {})])
def test_consistency_audit_accepts_real_runs(strategy, kw):
    from src.analysis.consistency import check
    d = run_single({**BASE, "strategy": strategy, "strategy_kwargs": kw,
                    "tag": "_cons_" + "_".join(f"{k}{v}" for k, v in kw.items())}, robust=False)
    fails, done = check(d)
    assert not fails and "schedule" in done


def test_quota_over_all_rounds_is_met():
    # rho = 0.25 over all 60 rounds with T_w = 5 full rounds leaves rho' = 10/55 after warm-up
    s = run_single({**BASE, "rounds": 30, "strategy": "tierfed", "tag": "_qall",
                    "strategy_kwargs": {"warmup": 3, "rho": 0.3, "quota": "all", "rounds": 30}}, robust=False)
    assert s["schedule"]["rho_post"] == pytest.approx((0.3 * 30 - 3) / 27)
    assert s["summary"]["realised_payload_fraction"] <= 0.3 + 0.02


def test_closed_form_matches_brute_force_and_bounds_the_gain():
    from scipy.optimize import minimize
    from src.theory.differentiation import closed_form, psi
    w, g, rho = {"a": 0.3, "b": 0.7}, {"a": 0.9, "b": 0.05}, 0.25
    cf = closed_form(w, g, rho)
    cons = {"type": "ineq", "fun": lambda K: rho - (0.3 / K[0] + 0.7 / K[1])}
    res = minimize(lambda K: K[0] ** 2 * 0.9 + K[1] ** 2 * 0.05, x0=[4.0, 4.0], constraints=[cons],
                   bounds=[(1e-3, None)] * 2, method="SLSQP")
    assert res.fun == pytest.approx(cf["J"], rel=1e-4)
    assert cf["K"]["a"] == pytest.approx(res.x[0], rel=1e-3) and cf["K"]["b"] == pytest.approx(res.x[1], rel=1e-3)
    # the integer optimum over K in {1..8}^2 never beats the uniform period by more than psi
    import itertools
    best = min(Ka ** 2 * 0.9 + Kb ** 2 * 0.05 for Ka, Kb in itertools.product(range(1, 9), repeat=2)
               if 0.3 / Ka + 0.7 / Kb <= rho + 1e-12)
    assert 1 - best / cf["J_uniform"] <= cf["psi"] + 1e-12
    assert psi({"a": 0.4, "b": 0.6}, {"a": 0.4, "b": 0.6}) == pytest.approx(0.0, abs=1e-12)


def test_exact_gaussian_accounting_is_tighter_than_rdp():
    from src.privacy.dp import epsilon_for
    from src.privacy.gdp import eps_gdp, sigma_gdp
    for T, s in ((18, 3.0), (60, 6.0), (1, 1.0)):
        e_rdp = epsilon_for(s, T, 1.0, 1e-5)
        e_gdp = eps_gdp(T, s, 1e-5)
        assert 0 < e_gdp <= e_rdp
    s = sigma_gdp(4.0, 60, 1e-5)
    assert eps_gdp(60, s, 1e-5) == pytest.approx(4.0, rel=1e-6)


def test_crossmachine_pair_comparison():
    import copy
    from src.analysis.crossmachine import compare_pair
    a = {"config": {"strategy": "fixed", "seed": 42, "device": "cpu"},
         "round_bits": [{"round": 1, "upload_bits": 64.0, "client_bits": [32.0, 32.0], "n_senders": 2.0,
                         "compute_s_max": 0.5}],
         "schedule": {"K_slow": 4.0, "K_medium": 4.0},
         "summary": {"valsel_round": 9.0, "fed_auroc_valsel": 0.62, "fed_auroc_final": 0.61, "total_upload_mb": 8e-6}}
    b = copy.deepcopy(a)
    b["config"]["device"] = "cuda"                  # the device is not part of the configuration
    b["round_bits"][0]["compute_s_max"] = 0.2       # wall-clock time is not part of the bits
    b["summary"]["fed_auroc_valsel"] = 0.625
    r = compare_pair(a, b)
    assert r["config_diff"] == [] and r["bits_same"] and r["K_same"] and r["sel_same"]
    assert r["d_auroc_valsel"] == pytest.approx(0.005)
    b["round_bits"][0]["client_bits"] = [32.0, 0.0]
    b["config"]["seed"] = 43
    r = compare_pair(a, b)
    assert not r["bits_same"] and r["config_diff"] == ["seed"]


def test_synchronisation_steps():
    """Restart (default): every upload is one local epoch away from the broadcast, so the
    dispersion of the uploads does not grow with the period. Carry-over
    (sync_restart=False): a tier is uploaded after all local epochs since its last
    synchronisation, so the dispersion grows with the period. A method that
    synchronises every round is the same run under both steps."""
    import pandas as pd
    base = {**BASE, "rounds": 14}

    def disp(res):
        h = pd.DataFrame(res["history"])
        tx = h[(h["round"] > 2) & (h["payload_fraction"] > 0)]
        return float(tx["divergence_medium"].mean())

    kw = {"k_slow": 4, "k_med": 4, "warmup": 2, "aligned": True}
    a = run_single({**base, "strategy": "fixed", "strategy_kwargs": kw, "tag": "_rs"}, robust=False)
    b = run_single({**base, "strategy": "fixed", "strategy_kwargs": kw, "tag": "_co", "sync_restart": False}, robust=False)
    assert a["summary"]["total_upload_mb"] == b["summary"]["total_upload_mb"]      # same schedule, same bytes
    assert a["summary"]["fed_auroc_final"] != b["summary"]["fed_auroc_final"]      # different trajectories
    assert disp(b) > 1.5 * disp(a)                                                  # uploads drift with the period
    f1 = run_single({**base, "strategy": "fedper", "tag": "_rs"}, robust=False)["summary"]
    f2 = run_single({**base, "strategy": "fedper", "tag": "_co", "sync_restart": False}, robust=False)["summary"]
    assert f1["fed_auroc_final"] == f2["fed_auroc_final"] and f1["total_upload_mb"] == f2["total_upload_mb"]
    with pytest.raises(ValueError):
        run_single({**base, "strategy": "fedper", "tag": "_co_p", "sync_restart": False, "participation": 0.5},
                   robust=False)

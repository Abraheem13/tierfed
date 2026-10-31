"""v4: subsampled accountant, sigma calibration, DP-safety guard, planner, loaders, emulator."""
import numpy as np
import pytest

from src.privacy.dp import calibrate_sigma, epsilon_for, private_aggregate
from src.strategies.adaptive import PrivacyPlannedTierFed


def test_accountant_reference_values():
    # TF-Privacy MNIST tutorial: q=256/60000, sigma=1.1, 60 epochs -> eps ~ 3.0 (delta 1e-5)
    assert abs(epsilon_for(1.1, 60 * 235, 256 / 60000) - 3.0) < 0.05
    # q = 1 reproduces the v3 (unsubsampled) accountant exactly
    assert abs(epsilon_for(1.0, 60, 1.0) - 67.188) < 1e-2


def test_subsampling_amplifies_and_calibration_is_tight():
    assert epsilon_for(3.0, 30, 0.5) < epsilon_for(3.0, 30, 1.0)
    for R in (8, 18, 60):
        s = calibrate_sigma(4.0, R, 0.5)
        assert epsilon_for(s, R, 0.5) <= 4.0 + 1e-9
        assert epsilon_for(s * 0.99, R, 0.5) > 4.0
    assert calibrate_sigma(4.0, 60, 1.0) > calibrate_sigma(4.0, 18, 1.0)


def test_empty_poisson_cohort_still_adds_noise():
    import torch
    g = {"w": torch.zeros(5)}
    out, _ = private_aggregate(g, [], ["w"], 1.0, 1.0, torch.Generator().manual_seed(0), denominator=2.0)
    assert float(out["w"].abs().sum()) > 0


def test_planner_is_data_free_and_meets_quota():
    names, sizes = ["a", "b", "c"], {"a": 1000, "b": 1000, "c": 100}
    tiers = {"a": "slow", "b": "medium", "c": "fast"}
    p = PrivacyPlannedTierFed(names, sizes, tiers, rounds=20, warmup=2, epsilon=4, q=0.5, rho=0.5, lam=1.0)
    assert p.dp_safe and p.planned_payload <= 0.5 + 1e-12
    assert len(p.planned_release_rounds(20)) == p.planned_releases
    hi = PrivacyPlannedTierFed(names, sizes, tiers, rounds=20, warmup=2, epsilon=4, q=0.5, lam=100.0)
    assert hi.planned_releases <= p.planned_releases or p.planned_payload < 0.5


def test_dp_guard_rejects_data_dependent_schedules():
    from src.experiments.runner import run_single
    with pytest.raises(ValueError):
        run_single(dict(dataset="synthetic", strategy="tierfed", rounds=2, n_clients=4, device="cpu",
                        out_dir="/tmp/nfl_guard", dp=dict(enabled=True, sigma=1.0)), robust=False)


def test_socket_emulator_matches_rate():
    from src.netemu import emulate_round
    t = emulate_round([250_000, 250_000], 20e6, 0.005)
    assert 0.9 * 0.105 < t < 0.105 * 1.3

"""Unit tests for tiering, strategies, metrics, compression, privacy, network."""
import warnings

import numpy as np
import pytest
import torch

warnings.filterwarnings("ignore")

from src.compression import build_compressor
from src.data import load_dataset
from src.data.partition import lpt_group_partition, dirichlet_label_skew
from src.losses import build_loss
from src.metrics import TrajectoryStats, classification_metrics
from src.models import assign_tiers, build_model, tier_keys, tier_report
from src.network import cost_run, PROFILES
from src.privacy import DPConfig, clip_update, private_aggregate, rdp_gaussian, rdp_to_dp
from src.strategies import build_strategy


def _mlp(in_dim=93):
    m = build_model("mlp", (in_dim,))
    t = assign_tiers(m)
    names = [n for n, _ in m.named_parameters()]
    sizes = {n: p.numel() for n, p in m.named_parameters()}
    return m, t, names, sizes


def test_tier_report_follows_input_dim():
    """The tier report must be computed from the model actually built, not a
    hard-coded figure (v2's paper quoted 93 inputs while runs used 162)."""
    for d in (93, 162):
        m, t, _, _ = _mlp(d)
        r = tier_report(m, t)
        assert r["total"]["params"] == d * 128 + 128 + 128 * 128 + 128 + 128 * 64 + 64 + 64 * 2 + 2
        assert abs(sum(r[k]["pct"] for k in ("slow", "medium", "fast")) - 100) < 1e-6


@pytest.mark.parametrize("name,shape,kw", [
    ("mlp", (93,), {}), ("resnet", (93,), {"depth": 3}),
    ("transformer", (93,), {"depth": 2}), ("cnn", (1, 28, 28), {}),
])
def test_tiering_generalises(name, shape, kw):
    m = build_model(name, shape, **kw)
    t = assign_tiers(m, example=torch.zeros(2, *shape))
    assert set(t.values()) <= {"slow", "medium", "fast"}
    assert len(t) == len(list(m.named_parameters()))
    for tier in ("slow", "medium", "fast"):
        assert len(tier_keys(t, tier)) > 0


def test_tierfed_exact_solver_respects_budget_and_beats_greedy():
    _, t, names, sizes = _mlp()
    head = tier_keys(t, "fast")
    for rho in (0.15, 0.25, 0.4, 0.6):
        ex = build_strategy("tierfed", names, sizes, tiers=t, head_keys=head, rho=rho, solver="exact")
        gr = build_strategy("tierfed", names, sizes, tiers=t, head_keys=head, rho=rho, solver="greedy")
        for s in (ex, gr):
            s.gamma = {"slow": 0.3, "medium": 1.0}
        Ke, Kg = ex._solve(), gr._solve()
        w = ex.tier_weight()
        assert sum(w[g] / Ke[g] for g in Ke) <= rho + 1e-9
        cost = lambda K: sum(ex.gamma[g] * K[g] ** 2 for g in K)  # noqa: E731
        assert cost(Ke) <= cost(Kg) + 1e-9
    # unequal divergence must produce tier-differentiated intervals at some budget
    ex = build_strategy("tierfed", names, sizes, tiers=t, head_keys=head, rho=0.4, solver="exact")
    ex.gamma = {"slow": 0.05, "medium": 1.0}
    K = ex._solve()
    assert K["slow"] > K["medium"]


def test_tierfed_local_only_rounds_and_private_head():
    _, t, names, sizes = _mlp()
    head = tier_keys(t, "fast")
    s = build_strategy("tierfed", names, sizes, tiers=t, head_keys=head, rho=0.25, warmup=2)
    assert all(k not in s.send_keys(r) for k in head for r in range(1, 30))
    s.gamma = {"slow": 1.0, "medium": 1.0}; s.K = {"slow": 4, "medium": 4}
    s.last_sync = {"slow": 3, "medium": 3}
    assert s.send_keys(4) == [] and s.send_keys(7) != []


def test_fixed_factorial_cells_distinct():
    _, t, names, sizes = _mlp()
    head = tier_keys(t, "fast")
    sched = build_strategy("fixed", names, sizes, tiers=t, head_keys=head, private_head=False, schedule=True)
    full = build_strategy("fixed", names, sizes, tiers=t, head_keys=head, private_head=True, schedule=True)
    assert sched.personal_keys() == [] and set(full.personal_keys()) == set(head)
    assert len({len(full.send_keys(r)) for r in range(1, 11)}) > 1
    # no fallback: with K_slow=5, K_med=2, round 3 has nothing due and is local-only
    sched = build_strategy("fixed", names, sizes, tiers=t, head_keys=head, k_slow=5, k_med=2)
    assert sched.send_keys(3) == []
    medium_rounds = sum(1 for r in range(1, 61) if any(t[k] == "medium" for k in sched.send_keys(r)))
    assert medium_rounds == 31            # r=1 plus the 30 even rounds


def test_fedlama_cumulative_share_rule():
    _, t, names, sizes = _mlp()
    s = build_strategy("fedlama", names, sizes, tau=2, phi=2, max_interval=8)
    g = {n: torch.zeros(v) for n, v in sizes.items()}
    cs = [{n: torch.randn(v) * (0.01 if "stem" in n else 1.0) for n, v in sizes.items()} for _ in range(4)]
    for r in range(1, 5):
        s.observe(g, cs, names, r)
    assert len(set(s.intervals.values())) > 1
    assert all(s.intervals[n] > 1 for n in names if "stem" in n)   # low-discrepancy, large layer -> long


def test_compression_acts_on_update_with_error_feedback():
    _, t, names, sizes = _mlp()
    s = build_strategy("fedavg", names, sizes, compressor=build_compressor("topk"))
    ref = {n: torch.randn(v) for n, v in sizes.items()}
    upd = {n: ref[n] + 0.01 * torch.randn(v) for n, v in sizes.items()}
    out, bits = s.compress_client_state(upd, names, reference=ref, cid=0)
    for n in names:   # reconstruction stays close to the client's weights, not a pruned weight vector
        assert (out[n] - upd[n]).abs().max() < 0.1
    assert bits < sum(v * 32 for v in sizes.values())
    assert s._residual[0]                              # error-feedback residual is kept


def test_scaffold_changes_training_objective():
    _, t, names, sizes = _mlp()
    s = build_strategy("scaffold", names, sizes)
    m = build_model("mlp", (93,))
    g = {n: p.detach().clone() for n, p in m.named_parameters()}
    s.c_global = {n: torch.ones_like(p) for n, p in m.named_parameters()}
    s.c_local[0] = {n: torch.zeros_like(p) for n, p in m.named_parameters()}
    base = torch.tensor(0.0, requires_grad=True)
    extra = s.local_objective_extra(m, g, base, cid=0)
    extra.backward()
    assert all(torch.allclose(p.grad, torch.ones_like(p)) for p in m.parameters())
    local = {n: v - 0.5 for n, v in g.items()}
    s.after_local(0, g, local, names, n_steps=10, lr=0.1)
    assert torch.allclose(s.c_local[0][names[0]], torch.full_like(g[names[0]], -1.0 + 0.5))
    assert s.upload_multiplier == 2.0


def test_lag_skips_small_changes():
    _, t, names, sizes = _mlp()
    s = build_strategy("lag", names, sizes, xi=1.0)
    g0 = {n: torch.zeros(v) for n, v in sizes.items()}
    c = {n: torch.ones(v) for n, v in sizes.items()}
    assert s.client_sends(0, g0, c, names, 1)          # first round always sends
    s.aggregate(g0, [c], [1.0], 1)
    g1 = {n: torch.ones(v) * 0.01 for n, v in sizes.items()}
    assert not s.client_sends(0, g1, {n: v + 0.01 for n, v in c.items()}, names, 2)


def test_adacomm_period_shrinks_with_loss():
    _, t, names, sizes = _mlp()
    s = build_strategy("adacomm", names, sizes, tau0=8)
    s.observe_loss(1, 1.0); assert s.tau == 8
    s.observe_loss(2, 0.25); assert s.tau == 4


def test_metrics_and_trajectory():
    rng = np.random.default_rng(0)
    y = (rng.random(400) < 0.11).astype(int)
    p = rng.random(400) * 0.3 + y * 0.3
    m = classification_metrics(y, p)
    for k in ("auroc", "auprc", "f1", "f1_best", "ece", "brier"):
        assert k in m and not np.isnan(m[k])
    assert np.isnan(classification_metrics(np.zeros(10, int), p[:10])["auroc"])
    s = TrajectoryStats("auroc", [0.60, 0.66, 0.64, 0.61, 0.58]).summary()
    assert abs(s["auroc_ppd"] - 0.08) < 1e-9 and "auroc_earlystop_value" in s


@pytest.mark.parametrize("name", ["none", "quant8", "quant4", "topk", "stc", "sketch"])
def test_compressors_report_bits(name):
    c = build_compressor(name)
    q, bits = c(torch.randn(500))
    assert q.shape == (500,) and bits > 0
    if name != "none":
        assert bits <= 500 * 32


def test_dp_per_client_clipping_and_empty_round():
    g = {"a": torch.zeros(50), "b": torch.zeros(10)}
    big = [{"a": torch.randn(50) * 10, "b": torch.randn(10) * 10} for _ in range(4)]
    out, info = private_aggregate(g, big, ["a", "b"], max_norm=1.0, sigma=0.0)
    delta = torch.cat([out["a"], out["b"]])
    assert delta.norm() <= 1.0 + 1e-5 and info["dp_clip_fraction"] == 1.0
    clipped, norm, f = clip_update({}, 1.0)                    # v2 crashed here
    assert clipped == {} and norm == 0.0 and f == 1.0
    out2, _ = private_aggregate(g, [], [], 1.0, 1.0)
    assert torch.equal(out2["a"], g["a"])


def test_dp_accountant_exact_gaussian():
    o, r = rdp_gaussian(1.0, 1)
    assert np.allclose(r, o / 2.0)
    cfg = DPConfig(enabled=True, sigma=1.0)
    assert cfg.epsilon(10) < cfg.epsilon(100)
    assert cfg.epsilon(0) == float("inf")
    # sigma=1, 1 step, delta=1e-5 -> epsilon ~ 4.9 (classic conversion bound)
    assert 4.0 < rdp_to_dp(o, r, 1e-5) < 6.0


@pytest.mark.parametrize("loss", ["weighted_ce", "focal", "ldam", "cb"])
def test_losses_backprop(loss):
    f = build_loss(loss, [900, 100], "cpu")
    logits = torch.randn(16, 2, requires_grad=True)
    out = f(logits, torch.randint(0, 2, (16,)))
    out.backward()
    assert torch.isfinite(out) and logits.grad is not None


def test_partition_cap_and_min_size():
    rng = np.random.default_rng(0)
    groups = np.array(["missing"] * 5000 + [f"s{i}" for i in rng.integers(0, 60, 5000)])
    parts = lpt_group_partition(groups, 20, max_share=0.15, seed=0)
    assert max(len(p) for p in parts) <= 0.15 * len(groups) + 200
    y = (rng.random(4000) < 0.1).astype(int)
    parts = dirichlet_label_skew(y, 20, 0.1, seed=0, min_size=30)
    assert min(len(p) for p in parts) >= 30


def test_grouped_client_split_has_no_patient_leakage():
    ds = load_dataset("synthetic", n_clients=6, n_per_client=120, seed=0, partition="natural")
    ds.groups = np.arange(len(ds.y_train)) // 3           # three rows per "patient"
    tr, te = ds.client_test_split(0.2, seed=1)
    for a, b in zip(tr, te):
        assert not set(ds.groups[a]) & set(ds.groups[b])


def test_network_cost_model():
    rb = [{"upload_bits": 8e6, "max_client_bits": 4e6, "n_senders": 2}, {"upload_bits": 0, "max_client_bits": 0, "n_senders": 0}]
    c = cost_run(rb, "nr_cell_edge")
    assert abs(c["net_nr_cell_edge_comm_time_s"] - (4e6 / 5e6 + 0.03)) < 1e-9
    assert c["net_nr_cell_edge_window_compliance"] == 1.0
    assert set(PROFILES) >= {"nr_cell_edge", "leo_sat", "sixg_slice"}


def test_partition_independent_of_hash_seed():
    """The specialty partition must not depend on PYTHONHASHSEED (v3 determinism bug)."""
    import json, os, subprocess, sys
    code = ("import numpy as np, json; from src.data.partition import lpt_group_partition, natural_sites;"
            "rng=np.random.default_rng(0); g=np.array([f'spec{i}' for i in rng.integers(0,40,3000)]);"
            "p=lpt_group_partition(g,20,max_share=0.15,seed=0); q=natural_sites(g,min_size=50,max_clients=10);"
            "print(json.dumps([[int(x) for x in a[:5]] for a in p]+[[int(x) for x in a[:5]] for a in q]))")
    outs = []
    for hs in ("1", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": hs}
        outs.append(subprocess.check_output([sys.executable, "-c", code], env=env, cwd=os.getcwd()).decode())
    assert outs[0] == outs[1]

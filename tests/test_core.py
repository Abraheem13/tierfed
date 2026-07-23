"""Unit tests for tiering, strategies, metrics, compression, privacy."""
import warnings

import numpy as np
import pytest
import torch

warnings.filterwarnings("ignore")

from src.compression import build_compressor
from src.data import load_dataset
from src.losses import build_loss
from src.metrics import TrajectoryStats, classification_metrics
from src.models import assign_tiers, build_model, tier_keys, tier_report
from src.privacy import DPConfig, clip_update
from src.strategies import build_strategy


def _mlp():
    m = build_model("mlp", (93,))
    t = assign_tiers(m)
    names = [n for n, _ in m.named_parameters()]
    sizes = {n: p.numel() for n, p in m.named_parameters()}
    return m, t, names, sizes


def test_mlp_tiering_matches_paper():
    m, t, _, _ = _mlp()
    r = tier_report(m, t)
    assert r["total"]["params"] == 36930
    assert abs(r["slow"]["pct"] - 32.6) < 0.2
    assert abs(r["medium"]["pct"] - 67.1) < 0.2
    assert abs(r["fast"]["pct"] - 0.35) < 0.05


@pytest.mark.parametrize("name,shape,kw", [
    ("mlp", (93,), {}), ("resnet", (93,), {"depth": 3}),
    ("transformer", (93,), {"depth": 2}), ("cnn", (1, 28, 28), {}),
])
def test_tiering_generalises(name, shape, kw):
    """Tiering must work beyond a two-hidden-layer MLP."""
    m = build_model(name, shape, **kw)
    t = assign_tiers(m, example=torch.zeros(2, *shape))
    assert set(t.values()) <= {"slow", "medium", "fast"}
    assert len(t) == len(list(m.named_parameters()))
    for tier in ("slow", "medium", "fast"):
        assert len(tier_keys(t, tier)) > 0, f"{name} has empty {tier} tier"


def test_factorial_2x2_distinct():
    """The four factorial cells must have genuinely different behaviour."""
    _, t, names, sizes = _mlp()
    head = tier_keys(t, "fast")
    cells = {
        "fedavg": build_strategy("fedavg", names, sizes, tiers=t, head_keys=head),
        "fedper": build_strategy("fedper", names, sizes, tiers=t, head_keys=head),
        "sched": build_strategy("nested", names, sizes, tiers=t, head_keys=head,
                                private_head=False, schedule=True),
        "nfl": build_strategy("nested", names, sizes, tiers=t, head_keys=head,
                              private_head=True, schedule=True),
    }
    assert cells["fedavg"].personal_keys() == []
    assert set(cells["fedper"].personal_keys()) == set(head)
    assert cells["sched"].personal_keys() == []
    assert set(cells["nfl"].personal_keys()) == set(head)
    # FedAvg/FedPer send a constant slice; scheduled variants vary by round.
    const = {len(cells["fedavg"].send_keys(r)) for r in range(1, 11)}
    varying = {len(cells["nfl"].send_keys(r)) for r in range(1, 11)}
    assert len(const) == 1 and len(varying) > 1


def test_nfl_schedule_and_head():
    _, t, names, sizes = _mlp()
    head = tier_keys(t, "fast")
    s = build_strategy("nested", names, sizes, tiers=t, head_keys=head, k_slow=5, k_med=2)
    assert all(k not in s.send_keys(r) for k in head for r in range(1, 21))
    assert len(s.send_keys(1)) == len(names) - len(head)      # round 1 syncs all shared
    assert len(s.send_keys(3)) > 0                            # fallback never empty
    assert s.payload_fraction(3) < 1.0


def test_fedlama_adapts_intervals():
    _, t, names, sizes = _mlp()
    s = build_strategy("fedlama", names, sizes, tau=2, max_interval=8)
    g = {n: torch.zeros(v) for n, v in sizes.items()}
    cs = [{n: torch.randn(v) * (0.01 if "stem" in n else 1.0) for n, v in sizes.items()}
          for _ in range(4)]
    for r in range(1, 5):
        s.observe_discrepancy(g, cs, r)
    assert len(set(s.intervals.values())) > 1, "FedLAMA must differentiate intervals"


def test_metrics_include_auprc_and_standard_stats():
    rng = np.random.default_rng(0)
    y = (rng.random(400) < 0.11).astype(int)
    p = rng.random(400) * 0.3 + y * 0.3
    m = classification_metrics(y, p)
    for k in ("auroc", "auprc", "auprc_lift", "f1", "f1_best", "ece", "brier"):
        assert k in m and not np.isnan(m[k])
    t = TrajectoryStats("auroc", [0.60, 0.66, 0.64, 0.61, 0.58])
    s = t.summary()
    assert abs(s["auroc_ppd"] - 0.08) < 1e-9
    assert "auroc_earlystop_value" in s and "auroc_auc_curve" in s


@pytest.mark.parametrize("name", ["none", "quant8", "quant4", "topk", "stc", "sketch"])
def test_compressors_report_bits(name):
    c = build_compressor(name)
    t = torch.randn(500)
    q, bits = c(t)
    assert q.shape == t.shape and bits > 0
    if name != "none":
        assert bits <= 500 * 32


def test_dp_clipping_and_epsilon():
    u = {"a": torch.randn(50) * 10}
    c, norm, f = clip_update(u, 1.0)
    assert float(torch.cat([v.flatten() for v in c.values()]).norm()) <= 1.0 + 1e-5
    cfg = DPConfig(enabled=True, sigma=1.0, sample_rate=0.5)
    assert cfg.epsilon(10) < cfg.epsilon(100)      # budget grows with rounds


@pytest.mark.parametrize("loss", ["weighted_ce", "focal", "ldam", "cb"])
def test_losses_backprop(loss):
    f = build_loss(loss, [900, 100], "cpu")
    logits = torch.randn(16, 2, requires_grad=True)
    out = f(logits, torch.randint(0, 2, (16,)))
    out.backward()
    assert torch.isfinite(out) and logits.grad is not None


def test_partition_schemes_report_heterogeneity():
    ds = load_dataset("synthetic", n_clients=6, n_per_client=120, seed=0, partition="dirichlet")
    rep = ds.meta["partition_report"]
    assert rep["n_clients"] == ds.n_clients
    assert "label_tv_mean" in rep and rep["label_tv_mean"] >= 0

"""End-to-end tests on the synthetic corpus: every strategy trains, DP is
applied to every strategy, repeated runs are bit-identical, attacks run."""
import json
import warnings
from pathlib import Path

import pytest

warnings.filterwarnings("ignore")

from src.experiments.runner import run_single

BASE = dict(dataset="synthetic", dataset_kwargs=dict(n_per_client=80, n_features=20),
            model="mlp", rounds=4, n_clients=4, seed=42, out_dir="/tmp/tierfed_test", device="cpu")
ALL = ["fedavg", "fedprox", "scaffold", "fedper", "fedlama", "adacomm", "lag", "fixed", "tierfed"]


@pytest.mark.parametrize("strategy", ALL)
def test_strategy_runs(strategy):
    res = run_single({**BASE, "strategy": strategy,
                      "strategy_kwargs": {"warmup": 1, "rho": 0.3} if strategy == "tierfed" else {}})
    s = res["summary"]
    for k in ("fed_auroc_final", "fed_auprc_final", "pooled_auroc_final", "total_upload_mb",
              "net_nr_cell_edge_comm_time_s", "realised_payload_fraction"):
        assert k in s
    assert len(res["history"]) == BASE["rounds"]
    assert s["total_upload_mb"] > 0


def test_determinism():
    a = run_single({**BASE, "strategy": "tierfed", "strategy_kwargs": {"warmup": 1, "rho": 0.3}})["summary"]
    b = run_single({**BASE, "strategy": "tierfed", "strategy_kwargs": {"warmup": 1, "rho": 0.3}})["summary"]
    for k in ("fed_auroc_final", "fed_auprc_final", "divergence_total_mean"):
        assert a[k] == b[k], f"{k} differs between identical runs"


@pytest.mark.parametrize("strategy", ["fedavg", "fedper", "tierfed"])
def test_dp_is_applied_to_every_strategy(strategy):
    kw = {"warmup": 1, "rho": 0.3} if strategy == "tierfed" else {}
    plain = run_single({**BASE, "strategy": strategy, "strategy_kwargs": kw})["summary"]
    noisy = run_single({**BASE, "strategy": strategy, "strategy_kwargs": kw,
                        "dp": dict(enabled=True, sigma=5.0, max_norm=0.1)})
    assert noisy["dp"]["dp_enabled"] == 1.0 and noisy["dp"]["dp_epsilon"] > 0
    assert noisy["dp"]["dp_rounds_charged"] == noisy["summary"]["rounds_transmitting"]
    assert noisy["summary"]["fed_auroc_final"] != plain["fed_auroc_final"], "DP had no effect"


def test_scaffold_differs_from_fedavg():
    a = run_single({**BASE, "strategy": "fedavg"})["summary"]
    b = run_single({**BASE, "strategy": "scaffold"})["summary"]
    assert a["fed_auroc_final"] != b["fed_auroc_final"]
    assert abs(b["total_upload_mb"] - 2 * a["total_upload_mb"]) < 1e-6


def test_tierfed_uploads_less_and_has_local_rounds():
    fp = run_single({**BASE, "strategy": "fedper", "rounds": 8})["summary"]
    tf = run_single({**BASE, "strategy": "tierfed", "rounds": 8, "strategy_kwargs": {"warmup": 2, "rho": 0.25}})["summary"]
    assert tf["total_upload_mb"] < fp["total_upload_mb"]
    assert tf["rounds_transmitting"] < 8


def test_attack_and_membership_run():
    res = run_single({**BASE, "strategy": "tierfed", "strategy_kwargs": {"warmup": 1, "rho": 0.3},
                      "attack_every": 1, "attack_steps": 5, "membership_inference": True})
    assert res["attack"] and "inversion_feature_corr" in res["attack"][0]
    assert "mia_auc_mean" in res["summary"]


def test_compression_reduces_upload():
    a = run_single({**BASE, "strategy": "fedavg"})["summary"]["total_upload_mb"]
    b = run_single({**BASE, "strategy": "fedavg", "compressor": "quant8"})["summary"]["total_upload_mb"]
    assert b < a


def test_collect_produces_tex():
    from src.analysis.collect import main as collect
    run_single({**BASE, "strategy": "fedper"})
    run_single({**BASE, "strategy": "tierfed", "strategy_kwargs": {"warmup": 1, "rho": 0.3}})
    collect(["--results", BASE["out_dir"], "--paper", BASE["out_dir"] + "/generated",
             "--dataset", "synthetic", "--model", "mlp"])
    gen = Path(BASE["out_dir"]) / "generated"
    assert (gen / "numbers.tex").exists() and (gen / "main_synthetic_mlp.tex").exists()
    assert "RUN REQUIRED" not in (gen / "main_synthetic_mlp.tex").read_text()
    assert "RUN REQUIRED" in (gen / "scaling.tex").read_text()

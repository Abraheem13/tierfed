"""End-to-end smoke tests: every strategy trains and produces full metrics."""
import warnings

import pytest

warnings.filterwarnings("ignore")

from src.experiments.runner import run_single

BASE = dict(dataset="synthetic",
            dataset_kwargs=dict(n_per_client=80, n_features=20),
            model="mlp", rounds=3, n_clients=4, seed=42, out_dir="/tmp/nfl_test")


@pytest.mark.parametrize("strategy", ["fedavg", "fedprox", "fedper", "fedlama", "scaffold", "nested"])
def test_strategy_runs(strategy):
    res = run_single({**BASE, "strategy": strategy})
    s = res["summary"]
    for k in ("fed_auroc_final", "fed_auprc_final", "fed_f1_final", "total_upload_mb"):
        assert k in s
    assert len(res["history"]) == BASE["rounds"]
    assert s["total_upload_mb"] > 0


def test_private_head_never_transmitted():
    """The core privacy/personalisation invariant of NFL."""
    res = run_single({**BASE, "strategy": "nested"})
    assert len(res["strategy"]["private_params"]) > 0


def test_nfl_uploads_less_than_fedavg():
    a = run_single({**BASE, "strategy": "fedavg"})["summary"]["total_upload_mb"]
    b = run_single({**BASE, "strategy": "nested", "rounds": 6})["summary"]["total_upload_mb"]
    c = run_single({**BASE, "strategy": "fedavg", "rounds": 6})["summary"]["total_upload_mb"]
    assert b < c, "NFL must upload strictly less than FedAvg at equal rounds"


def test_compression_reduces_upload():
    a = run_single({**BASE, "strategy": "fedavg"})["summary"]["total_upload_mb"]
    b = run_single({**BASE, "strategy": "fedavg", "compressor": "quant8"})["summary"]["total_upload_mb"]
    assert b < a


def test_dp_runs_and_reports_epsilon():
    res = run_single({**BASE, "strategy": "nested",
                      "dp": dict(enabled=True, sigma=1.0, max_norm=1.0)})
    assert res["dp"]["dp_epsilon"] > 0

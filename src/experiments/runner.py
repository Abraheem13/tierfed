"""Single-run driver shared by every experiment script."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import torch

from ..compression import build_compressor
from ..data import load_dataset
from ..models import assign_tiers, build_model, tier_keys, tier_report
from ..privacy.dp import DPConfig
from ..server import FederatedServer
from ..strategies import build_strategy
from ..utils import get_device, get_logger, save_json, set_seed

log = get_logger()

DEFAULTS = dict(
    dataset="diabetes", dataset_kwargs=None, model="mlp", model_kwargs=None,
    strategy="nested", strategy_kwargs=None, rounds=60, local_epochs=1,
    batch_size=64, lr=1e-3, loss="weighted_ce", balanced_sampler=False,
    n_clients=20, participation=1.0, seed=42, device="auto",
    compressor=None, compressor_kwargs=None, dp=None,
    eval_every=1, early_stop_patience=10, client_test_frac=0.2, out_dir="results",
)


def run_single(cfg: Dict) -> Dict:
    cfg = {**DEFAULTS, **(cfg or {})}
    set_seed(cfg["seed"])
    device = get_device(cfg["device"])

    ds = load_dataset(cfg["dataset"], n_clients=cfg["n_clients"], seed=cfg["seed"],
                      **(cfg.get("dataset_kwargs") or {}))
    mkw = cfg.get("model_kwargs") or {}
    model_fn = lambda: build_model(cfg["model"], ds.input_shape, 2, **mkw)  # noqa: E731

    probe = model_fn()
    example = torch.zeros(2, *ds.input_shape)
    tiers = assign_tiers(probe, example=example)
    names = [n for n, _ in probe.named_parameters()]
    sizes = {n: p.numel() for n, p in probe.named_parameters()}
    head = tier_keys(tiers, "fast")

    skw = dict(cfg.get("strategy_kwargs") or {})
    skw["compressor"] = build_compressor(cfg.get("compressor"), **(cfg.get("compressor_kwargs") or {}))
    dpc = cfg.get("dp")
    skw["dp"] = DPConfig(**dpc) if isinstance(dpc, dict) else (dpc or DPConfig())
    strategy = build_strategy(cfg["strategy"], names, sizes, tiers=tiers, head_keys=head, **skw)

    server = FederatedServer(ds, model_fn, strategy, device, cfg)
    server.divergence.set_tiers(tiers)
    log.info(f"{cfg['strategy']:>12s} | {cfg['dataset']}/{cfg['model']} | seed {cfg['seed']} "
             f"| {ds.n_clients} clients | {cfg['rounds']} rounds | device {device}")
    res = server.run()

    res["config"] = {k: v for k, v in cfg.items() if k != "dataset_kwargs"}
    res["dataset"] = ds.describe()
    res["tier_report"] = tier_report(probe, tiers)
    res["dp"] = skw["dp"].summary(cfg["rounds"])
    if hasattr(strategy, "schedule_table"):
        res["schedule"] = strategy.schedule_table(cfg["rounds"])

    tag = (f"{cfg['dataset']}_{cfg['model']}_{cfg['strategy']}_R{cfg['rounds']}"
           f"_C{ds.n_clients}_s{cfg['seed']}{cfg.get('tag','')}")
    out = Path(cfg["out_dir"])
    (out / "logs").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(res["history"]).to_csv(out / "logs" / f"{tag}_history.csv", index=False)
    save_json(out / "logs" / f"{tag}_summary.json",
              {k: v for k, v in res.items() if k != "history"})
    s = res["summary"]
    log.info(f"  -> AUROC final {s.get('fed_auroc_final', float('nan')):.4f} "
             f"best {s.get('fed_auroc_best', float('nan')):.4f} "
             f"PPD {s.get('fed_auroc_ppd', float('nan')):.4f} | "
             f"AUPRC {s.get('fed_auprc_final', float('nan')):.4f} | "
             f"upload {s.get('total_upload_mb', float('nan')):.1f} MB")
    return res

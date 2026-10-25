"""Single-run driver shared by every experiment script."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
import torch

from ..compression import build_compressor
from ..data import load_dataset
from ..models import assign_tiers, build_model, tier_keys, tier_report
from ..network import cost_all_profiles
from ..privacy.dp import DPConfig
from ..server import FederatedServer
from ..strategies import build_strategy
from ..utils import get_device, get_logger, save_json, set_seed

log = get_logger()

# One architecture table shared by EVERY strategy, so sweeps are like-for-like.
ARCH = {
    "mlp": dict(hidden=(128, 128, 64), dropout=0.3),
    "resnet": dict(dim=128, depth=6, dropout=0.2),
    "transformer": dict(dim=64, depth=3, heads=8, dropout=0.1),
    "cnn": dict(width=32, dropout=0.2),
}

DEFAULTS = dict(
    dataset="diabetes", dataset_kwargs=None, model="mlp", model_kwargs=None,
    strategy="tierfed", strategy_kwargs=None, rounds=60, local_epochs=1,
    batch_size=64, lr=1e-3, optimizer="adam", loss="weighted_ce", balanced_sampler=False,
    n_clients=20, participation=1.0, seed=42, device="auto",
    compressor=None, compressor_kwargs=None, dp=None,
    eval_every=1, early_stop_patience=10, client_test_frac=0.2, out_dir="results",
    attack_every=0, attack_steps=300, attack_batch=8, membership_inference=False, tag="",
)


def run_tag(cfg: Dict, n_clients: int) -> str:
    tag = cfg.get("tag", "") or ""
    dpc = cfg.get("dp")
    sigma = dpc.get("sigma") if isinstance(dpc, dict) and dpc.get("enabled") else getattr(dpc, "sigma", None) if dpc and getattr(dpc, "enabled", False) else None
    if sigma and "dp" not in tag:
        tag += f"_dp{sigma}"
    if cfg.get("compressor") and "comp" not in tag:
        tag += f"_comp_{cfg['compressor']}"
    return (f"{cfg['dataset']}_{cfg['model']}_{cfg['strategy']}_R{cfg['rounds']}"
            f"_C{n_clients}_s{cfg['seed']}{tag}")


def run_single(cfg: Dict, skip_existing: bool = False) -> Dict:
    cfg = {**DEFAULTS, **(cfg or {})}
    set_seed(cfg["seed"])
    device = get_device(cfg["device"])

    ds = load_dataset(cfg["dataset"], n_clients=cfg["n_clients"], seed=cfg["seed"],
                      **(cfg.get("dataset_kwargs") or {}))
    out = Path(cfg["out_dir"]); (out / "logs").mkdir(parents=True, exist_ok=True)
    tag = run_tag(cfg, ds.n_clients)
    if skip_existing and (out / "logs" / f"{tag}_summary.json").exists():
        log.info(f"skip existing {tag}")
        import json
        return json.loads((out / "logs" / f"{tag}_summary.json").read_text())

    mkw = {**ARCH.get(cfg["model"], {}), **(cfg.get("model_kwargs") or {})}
    model_fn = lambda: build_model(cfg["model"], ds.input_shape, 2, **mkw)  # noqa: E731

    probe = model_fn()
    example = torch.zeros(2, *ds.input_shape)
    tiers = assign_tiers(probe, example=example)
    names = [n for n, _ in probe.named_parameters()]
    sizes = {n: p.numel() for n, p in probe.named_parameters()}
    head = tier_keys(tiers, "fast")

    skw = dict(cfg.get("strategy_kwargs") or {})
    skw["compressor"] = build_compressor(cfg.get("compressor"), **(cfg.get("compressor_kwargs") or {}))
    strategy = build_strategy(cfg["strategy"], names, sizes, tiers=tiers, head_keys=head, **skw)
    dpc = cfg.get("dp")
    dp = DPConfig(**dpc) if isinstance(dpc, dict) else (dpc or DPConfig())

    server = FederatedServer(ds, model_fn, strategy, device, cfg, dp=dp)
    server.divergence.set_tiers(tiers)
    log.info(f"{cfg['strategy']:>10s} | {cfg['dataset']}/{cfg['model']} | seed {cfg['seed']} "
             f"| {ds.n_clients} clients | {cfg['rounds']} rounds | device {device} | {tag}")
    res = server.run()

    res["config"] = {k: v for k, v in cfg.items()}
    res["config"]["tag"] = tag.split(f"_s{cfg['seed']}", 1)[1]
    res["config"]["model_kwargs_effective"] = mkw
    res["dataset"] = ds.describe()
    res["tier_report"] = tier_report(probe, tiers)
    res["tiers"] = tiers
    res["network"] = cost_all_profiles(res["round_bits"])
    res["summary"].update(res["network"])
    if hasattr(strategy, "schedule_table"):
        res["schedule"] = strategy.schedule_table(cfg["rounds"])

    pd.DataFrame(res["history"]).to_csv(out / "logs" / f"{tag}_history.csv", index=False)
    save_json(out / "logs" / f"{tag}_summary.json",
              {k: v for k, v in res.items() if k != "history"})
    s = res["summary"]
    log.info(f"  -> AUROC final {s.get('fed_auroc_final', float('nan')):.4f} "
             f"best {s.get('fed_auroc_best', float('nan')):.4f} | AUPRC {s.get('fed_auprc_final', float('nan')):.4f} "
             f"| upload {s.get('total_upload_mb', float('nan')):.1f} MB | tx rounds {s.get('rounds_transmitting', 0):.0f}")
    return res

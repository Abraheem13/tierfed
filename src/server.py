"""Federated simulation engine.

Self-contained (no Flower dependency) so that tier-wise communication,
compression, DP noise and per-round diagnostics are all measured exactly and
runs are fast enough for 10+ seeds across many configurations on one GPU.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from .client import Client
from .metrics import TrajectoryStats, classification_metrics, weighted_average
from .theory.divergence import DivergenceTracker
from .utils import clone_state, get_logger

log = get_logger()


class FederatedServer:
    def __init__(self, dataset, model_fn: Callable, strategy, device, cfg: Dict):
        self.dataset, self.model_fn, self.strategy = dataset, model_fn, strategy
        self.device, self.cfg = device, cfg
        self.global_model = model_fn().to(device)
        self.global_state = clone_state(self.global_model.state_dict())
        self.clients = [Client(k, dataset, model_fn, device, cfg) for k in range(dataset.n_clients)]
        self.rounds = cfg.get("rounds", 60)
        self.local_epochs = cfg.get("local_epochs", 1)
        self.participation = cfg.get("participation", 1.0)
        self.seed = cfg.get("seed", 42)
        self.history: List[Dict[str, float]] = []
        self.upload_bits = 0.0
        self.download_bits = 0.0
        self.divergence = DivergenceTracker()
        # Personalised evaluation needs per-client held-out data.
        self.local_tr, self.local_te = dataset.client_test_split(
            frac=cfg.get("client_test_frac", 0.2), seed=self.seed)

    # ------------------------------------------------------------------ #
    def _sample(self, r: int) -> List[int]:
        n = len(self.clients)
        m = max(1, int(round(self.participation * n)))
        if m >= n:
            return list(range(n))
        rng = np.random.default_rng(self.seed * 1000 + r)
        return sorted(rng.choice(n, size=m, replace=False).tolist())

    def run(self) -> Dict:
        priv = self.strategy.personal_keys()
        for r in range(1, self.rounds + 1):
            keys = self.strategy.send_keys(r)
            sel = self._sample(r)
            payload = {k: self.global_state[k] for k in keys}
            self.download_bits += len(sel) * sum(v.numel() * 32 for v in payload.values())

            states, weights, stats = [], [], []
            for cid in sel:
                c = self.clients[cid]
                c.set_parameters(self.global_state, keys)
                s = c.train(self.strategy, payload if self.cfg.get("needs_global", True) else None,
                            self.local_epochs, self.seed)
                c.stash_private(priv)
                cs = c.get_slice(keys)
                cs = self.strategy.compress_client_state(cs, keys)
                states.append(cs); weights.append(c.n); stats.append(s)

            self.upload_bits += self.strategy.measure_upload(states, keys)
            self.global_state = self.strategy.aggregate(self.global_state, states, weights, r)
            self.divergence.update(r, self.global_state, states, keys)

            if r % self.cfg.get("eval_every", 1) == 0 or r == self.rounds:
                self.history.append(self._evaluate(r, stats, keys))
        return self.summarise()

    # ------------------------------------------------------------------ #
    def _evaluate(self, r: int, stats, keys) -> Dict[str, float]:
        """Federated evaluation: each client evaluates on its OWN held-out split
        using its own personalised parameters, then results are sample-weighted.
        This is the only correct protocol for personalised methods."""
        per_client, weights = [], []
        xte_all, yte_all, p_all = [], [], []
        for cid, c in enumerate(self.clients):
            idx = self.local_te[cid]
            if len(idx) == 0:
                continue
            c.set_parameters(self.global_state, keys)
            x = torch.as_tensor(self.dataset.x_train[idx], dtype=torch.float32)
            y = self.dataset.y_train[idx]
            p = c.predict(x)
            per_client.append(classification_metrics(y, p))
            weights.append(len(idx))
            xte_all.append(x); yte_all.append(y); p_all.append(p)
        fed = weighted_average(per_client, weights)

        # Global (non-personalised) view on the untouched held-out test set.
        self.global_model.load_state_dict(self.global_state)
        self.global_model.eval()
        with torch.no_grad():
            xt = torch.as_tensor(self.dataset.x_test, dtype=torch.float32).to(self.device)
            pg = torch.softmax(self.global_model(xt), 1)[:, 1].cpu().numpy()
        glob = classification_metrics(self.dataset.y_test, pg)

        row = {"round": float(r),
               "train_loss": float(np.mean([s["loss"] for s in stats])) if stats else np.nan,
               "grad_sq": float(np.mean([s["grad_sq"] for s in stats])) if stats else np.nan,
               "upload_mb": self.upload_bits / 8e6, "download_mb": self.download_bits / 8e6,
               "payload_fraction": float(sum(self.strategy.sizes[k] for k in keys) /
                                         max(sum(self.strategy.sizes.values()), 1))}
        row.update({f"fed_{k}": v for k, v in fed.items()})
        row.update({f"global_{k}": v for k, v in glob.items()})
        row.update(self.divergence.latest())
        return row

    def summarise(self) -> Dict:
        out: Dict[str, float] = {}
        for m in ("fed_auroc", "fed_auprc", "fed_f1", "fed_f1_best", "fed_ece",
                  "global_auroc", "global_auprc", "global_f1"):
            vals = [h[m] for h in self.history if m in h and not np.isnan(h[m])]
            if vals:
                out.update(TrajectoryStats(m, vals).summary(
                    patience=self.cfg.get("early_stop_patience", 10)))
        out["total_upload_mb"] = self.upload_bits / 8e6
        out["total_download_mb"] = self.download_bits / 8e6
        out["rounds"] = float(self.rounds)
        out.update(self.divergence.summary())
        return {"summary": out, "history": self.history,
                "strategy": self.strategy.describe()}

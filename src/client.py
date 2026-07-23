"""Local client training."""
from __future__ import annotations

from typing import Dict, List, Mapping, Optional

import numpy as np
import torch
import torch.nn as nn

from .losses import build_loss
from .utils import clone_state


class Client:
    def __init__(self, cid: int, dataset, model_fn, device, cfg):
        self.cid, self.device, self.cfg = cid, device, cfg
        self.dataset = dataset
        self.model = model_fn().to(device)
        self.counts = dataset.client_counts(cid)
        self.n = int(sum(self.counts))
        self.loss_fn = build_loss(cfg.get("loss", "weighted_ce"), self.counts, device,
                                  focal_gamma=cfg.get("focal_gamma", 2.0))
        self.personal: Dict[str, torch.Tensor] = {}
        self.opt_state = None
        self._loader = None

    # ------------------------------------------------------------------ #
    def loader(self, seed: int):
        if self._loader is None:
            self._loader = self.dataset.client_loader(
                self.cid, batch_size=self.cfg.get("batch_size", 64),
                balanced=self.cfg.get("balanced_sampler", False), seed=seed)
        return self._loader

    def set_parameters(self, state: Mapping[str, torch.Tensor], keys: List[str]):
        """Load the broadcast slice, then restore this client's private tensors."""
        cur = self.model.state_dict()
        for k in keys:
            if k in state:
                cur[k] = state[k].to(self.device)
        for k, v in self.personal.items():
            cur[k] = v.to(self.device)
        self.model.load_state_dict(cur)

    def get_slice(self, keys: List[str]) -> Dict[str, torch.Tensor]:
        sd = self.model.state_dict()
        return {k: sd[k].detach().clone() for k in keys if k in sd}

    def stash_private(self, private_keys: List[str]):
        sd = self.model.state_dict()
        for k in private_keys:
            if k in sd:
                self.personal[k] = sd[k].detach().clone()

    # ------------------------------------------------------------------ #
    def train(self, strategy, global_params: Optional[Dict[str, torch.Tensor]],
              epochs: int, seed: int) -> Dict[str, float]:
        self.model.train()
        opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg.get("lr", 1e-3))
        if self.opt_state is not None and self.cfg.get("persist_optimizer", True):
            try:
                opt.load_state_dict(self.opt_state)
            except Exception:
                pass
        total, nb = 0.0, 0
        grad_sq = 0.0
        for _ in range(epochs):
            for xb, yb in self.loader(seed):
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                loss = self.loss_fn(self.model(xb), yb)
                loss = strategy.local_objective_extra(self.model, global_params, loss)
                loss.backward()
                if self.cfg.get("grad_clip", 0) > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["grad_clip"])
                gs = sum(float((p.grad ** 2).sum()) for p in self.model.parameters() if p.grad is not None)
                grad_sq += gs
                opt.step()
                total += float(loss.detach()); nb += 1
        self.opt_state = opt.state_dict()
        return {"loss": total / max(nb, 1), "grad_sq": grad_sq / max(nb, 1), "n": self.n}

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> np.ndarray:
        self.model.eval()
        p = torch.softmax(self.model(x.to(self.device)), dim=1)[:, 1]
        return p.cpu().numpy()

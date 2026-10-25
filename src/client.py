"""Local client training."""
from __future__ import annotations

from typing import Dict, List, Mapping, Optional

import numpy as np
import torch
import torch.nn as nn

from .losses import build_loss


class Client:
    def __init__(self, cid: int, dataset, model_fn, device, cfg, train_idx=None):
        self.cid, self.device, self.cfg = cid, device, cfg
        self.dataset = dataset
        self.model = model_fn().to(device)
        self.train_idx = train_idx if train_idx is not None else dataset.client_indices[cid]
        y = dataset.y_train[self.train_idx]
        self.counts = [int((y == c).sum()) for c in (0, 1)]
        self.n = int(len(self.train_idx))
        self.loss_fn = build_loss(cfg.get("loss", "weighted_ce"), self.counts, device,
                                  focal_gamma=cfg.get("focal_gamma", 2.0))
        self.personal: Dict[str, torch.Tensor] = {}
        self.opt_state = None
        self._loader = None

    # ------------------------------------------------------------------ #
    def loader(self, seed: int):
        if self._loader is None:
            self._loader = self.dataset.loader_for(
                self.train_idx, batch_size=self.cfg.get("batch_size", 64),
                balanced=self.cfg.get("balanced_sampler", False), seed=seed + self.cid)
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
        return {k: sd[k].detach().clone().cpu() for k in keys if k in sd}

    def stash_private(self, private_keys: List[str]):
        sd = self.model.state_dict()
        for k in private_keys:
            if k in sd:
                self.personal[k] = sd[k].detach().clone()

    # ------------------------------------------------------------------ #
    def train(self, strategy, global_params: Optional[Dict[str, torch.Tensor]],
              epochs: int, seed: int) -> Dict[str, float]:
        self.model.train()
        lr = self.cfg.get("lr", 1e-3)
        opt_name = self.cfg.get("optimizer", "adam")
        if opt_name == "sgd":
            opt = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=self.cfg.get("momentum", 0.0))
        else:
            opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        if self.opt_state is not None and self.cfg.get("persist_optimizer", True):
            try:
                opt.load_state_dict(self.opt_state)
            except Exception:
                pass
        if global_params is not None:
            global_params = {k: v.to(self.device) for k, v in global_params.items()}
        total, nb, grad_sq = 0.0, 0, 0.0
        for _ in range(epochs):
            for xb, yb in self.loader(seed):
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                loss = self.loss_fn(self.model(xb), yb)
                loss = strategy.local_objective_extra(self.model, global_params, loss, cid=self.cid)
                loss.backward()
                if self.cfg.get("grad_clip", 0) > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["grad_clip"])
                grad_sq += sum(float((p.grad ** 2).sum()) for p in self.model.parameters()
                               if p.grad is not None)
                opt.step()
                total += float(loss.detach()); nb += 1
        self.opt_state = opt.state_dict()
        return {"loss": total / max(nb, 1), "grad_sq": grad_sq / max(nb, 1),
                "n": self.n, "steps": nb, "lr": lr}

    @torch.no_grad()
    def predict(self, x: torch.Tensor, batch: int = 4096) -> np.ndarray:
        self.model.eval()
        outs = []
        for i in range(0, len(x), batch):
            outs.append(torch.softmax(self.model(x[i:i + batch].to(self.device)), dim=1)[:, 1].cpu())
        return torch.cat(outs).numpy() if outs else np.zeros(0)

    @torch.no_grad()
    def per_sample_loss(self, x: torch.Tensor, y: torch.Tensor, batch: int = 4096) -> np.ndarray:
        self.model.eval()
        outs = []
        for i in range(0, len(x), batch):
            logits = self.model(x[i:i + batch].to(self.device))
            outs.append(nn.functional.cross_entropy(logits, y[i:i + batch].to(self.device),
                                                    reduction="none").cpu())
        return torch.cat(outs).numpy() if outs else np.zeros(0)

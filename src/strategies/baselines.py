"""Baseline strategies, including the two whose absence reviewers identified
as disqualifying: FedPer and FedLAMA."""
from __future__ import annotations

from typing import Dict, List, Mapping, Sequence

import numpy as np
import torch

from .base import Strategy


class FedAvg(Strategy):
    """McMahan et al. -- sample-weighted average of every parameter each round."""
    name = "fedavg"


class FedProx(Strategy):
    """Li et al. -- FedAvg plus a proximal term in the local objective."""
    name = "fedprox"

    def __init__(self, *a, mu: float = 0.01, **kw):
        super().__init__(*a, **kw)
        self.mu = mu

    def local_objective_extra(self, model, global_params, batch_loss):
        if self.mu <= 0 or global_params is None:
            return batch_loss
        prox = 0.0
        for n, p in model.named_parameters():
            if n in global_params:
                prox = prox + ((p - global_params[n].to(p.device)) ** 2).sum()
        return batch_loss + 0.5 * self.mu * prox

    def describe(self):
        return {**super().describe(), "mu": self.mu}


class FedPer(Strategy):
    """Arivazhagan et al. -- shared backbone, private classification head.

    This is the decisive ablation the reviewers demanded: it keeps the head
    private (like NFL) but synchronises every shared layer EVERY round (unlike
    NFL). Comparing FedPer with NFL therefore isolates the multi-frequency
    schedule from the personalisation head.
    """
    name = "fedper"

    def __init__(self, param_names, sizes, head_keys: Sequence[str], **kw):
        super().__init__(param_names, sizes, **kw)
        self.private = list(head_keys)

    def describe(self):
        return {**super().describe(), "n_private": len(self.private)}


class FedLAMA(Strategy):
    """Lee et al. -- layer-wise adaptive model aggregation.

    Faithful to the published mechanism: every `tau` rounds the server measures
    per-layer model discrepancy across clients, sorts layers, and assigns each
    an aggregation interval from a geometric ladder so that low-discrepancy
    layers synchronise less often. Intervals are recomputed adaptively, which
    is the substantive difference from NFL's fixed semantic partition.
    """
    name = "fedlama"

    def __init__(self, param_names, sizes, tau: int = 5, phi: int = 2,
                 max_interval: int = 8, **kw):
        super().__init__(param_names, sizes, **kw)
        self.tau, self.phi, self.max_interval = tau, phi, max_interval
        self.intervals: Dict[str, int] = {k: 1 for k in self.param_names}
        self._disc: Dict[str, float] = {}

    def send_keys(self, r: int) -> List[str]:
        if r <= 1:
            return list(self.param_names)
        return [k for k in self.param_names if r % max(1, self.intervals.get(k, 1)) == 0]

    def observe_discrepancy(self, global_state, client_states, r: int):
        """Layer-wise discrepancy delta_l = mean_k ||theta_l^k - theta_l_bar||."""
        if not client_states:
            return
        for k in self.param_names:
            vals = [cs[k].float() for cs in client_states if k in cs]
            if len(vals) < 2:
                continue
            mean = torch.stack(vals).mean(0)
            d = float(torch.stack([(v - mean).norm() for v in vals]).mean())
            scale = float(mean.norm()) or 1.0
            self._disc[k] = d / scale
        if r % self.tau == 0 and self._disc:
            self._reassign()

    def _reassign(self):
        """Assign intervals: small discrepancy -> long interval."""
        keys = [k for k in self.param_names if k in self._disc]
        order = sorted(keys, key=lambda k: self._disc[k])       # ascending discrepancy
        n = len(order)
        if n == 0:
            return
        ladder = []
        v = self.max_interval
        while v >= 1:
            ladder.append(int(v)); v //= self.phi
        ladder = ladder or [1]
        per = max(1, int(np.ceil(n / len(ladder))))
        for i, k in enumerate(order):
            self.intervals[k] = ladder[min(i // per, len(ladder) - 1)]

    def aggregate(self, global_state, client_states, weights, r):
        self.observe_discrepancy(global_state, client_states, r)
        return super().aggregate(global_state, client_states, weights, r)

    def describe(self):
        return {**super().describe(), "tau": self.tau, "phi": self.phi,
                "max_interval": self.max_interval,
                "mean_interval": float(np.mean(list(self.intervals.values())))}


class SCAFFOLD(Strategy):
    """Karimireddy et al. control-variate correction (extra drift baseline)."""
    name = "scaffold"

    def __init__(self, param_names, sizes, **kw):
        super().__init__(param_names, sizes, **kw)
        self.c_global: Dict[str, torch.Tensor] = {}
        self.c_local: Dict[int, Dict[str, torch.Tensor]] = {}

    def measure_upload(self, client_states, keys) -> float:
        # SCAFFOLD transmits the update AND the control variate: double payload.
        return 2.0 * super().measure_upload(client_states, keys)

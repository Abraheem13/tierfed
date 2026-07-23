"""Client-level differential privacy for federated updates.

Answers the objection that the manuscript claimed GDPR relevance without any
privacy mechanism or evaluation. Implements update clipping + Gaussian noise
(the standard client-level DP-FedAvg mechanism of McMahan et al.) with an RDP
accountant so an (epsilon, delta) figure can be reported per configuration.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping

import numpy as np
import torch


def clip_update(update: Mapping[str, torch.Tensor], max_norm: float):
    """Clip the flattened update to an L2 ball of radius `max_norm`."""
    total = torch.sqrt(sum((v.detach().float() ** 2).sum() for v in update.values()))
    factor = min(1.0, max_norm / float(total.clamp_min(1e-12)))
    return {k: v * factor for k, v in update.items()}, float(total), factor


def add_gaussian_noise(update: Mapping[str, torch.Tensor], sigma: float, max_norm: float,
                       n_clients: int, generator: torch.Generator | None = None):
    """Server-side Gaussian mechanism on the averaged update."""
    std = sigma * max_norm / max(n_clients, 1)
    out = {}
    for k, v in update.items():
        noise = torch.normal(0.0, std, size=v.shape, generator=generator,
                             device=v.device, dtype=v.dtype) if std > 0 else 0.0
        out[k] = v + noise
    return out


def rdp_gaussian(sigma: float, q: float, steps: int, orders=None):
    """RDP of the sampled Gaussian mechanism (Mironov et al. bound)."""
    if orders is None:
        orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
    rdp = []
    for a in orders:
        if sigma <= 0:
            rdp.append(float("inf")); continue
        # Standard upper bound for the subsampled Gaussian mechanism.
        eps_a = q ** 2 * a / (sigma ** 2)
        rdp.append(eps_a * steps)
    return np.array(orders, dtype=float), np.array(rdp, dtype=float)


def rdp_to_dp(orders: np.ndarray, rdp: np.ndarray, delta: float = 1e-5) -> float:
    """Convert RDP to (eps, delta)-DP, minimising over orders."""
    with np.errstate(over="ignore", invalid="ignore"):
        eps = rdp + np.log1p(-1.0 / orders) - (np.log(delta) + np.log(orders)) / (orders - 1.0)
    eps = eps[np.isfinite(eps)]
    return float(np.min(eps)) if eps.size else float("inf")


@dataclass
class DPConfig:
    enabled: bool = False
    max_norm: float = 1.0
    sigma: float = 1.0           # noise multiplier
    delta: float = 1e-5
    sample_rate: float = 1.0     # client sampling fraction q

    def epsilon(self, rounds: int) -> float:
        if not self.enabled or self.sigma <= 0:
            return float("inf")
        o, r = rdp_gaussian(self.sigma, self.sample_rate, rounds)
        return rdp_to_dp(o, r, self.delta)

    def summary(self, rounds: int) -> Dict[str, float]:
        return {"dp_enabled": float(self.enabled), "dp_sigma": self.sigma,
                "dp_max_norm": self.max_norm, "dp_delta": self.delta,
                "dp_epsilon": self.epsilon(rounds)}

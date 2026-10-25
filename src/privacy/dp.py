"""Client-level differential privacy (DP-FedAvg, McMahan et al. 2018).

v2 clipped the *aggregated* update and added noise scaled by C/|S|. That is
not client-level DP: replacing one client can move the un-clipped aggregate
arbitrarily, so the sensitivity argument behind the reported (epsilon, delta)
did not hold. v2 also never applied the mechanism to FedAvg/FedPer/FedProx and
crashed for TierFed on local-only rounds.

v3 mechanism (applied by the server for EVERY strategy, see server.py):

    delta_k   = theta_k - theta           (the client's transmitted slice)
    delta_k  <- delta_k * min(1, C / ||delta_k||_2)          per-client clip
    sum       = sum_k delta_k + N(0, sigma^2 C^2 I)         Gaussian mechanism
    theta'    = theta + sum / |S|                           uniform average

Uniform (not sample-weighted) averaging is used under DP so the L2 sensitivity
of `sum` to one client is exactly C. Rounds in which a strategy transmits
nothing consume no privacy budget. The accountant is the exact Renyi DP of the
Gaussian mechanism, alpha/(2 sigma^2) per round, composed over the rounds in
which something was transmitted, converted with the standard bound of
Mironov (2017). Full participation (q = 1) is assumed, so no subsampling
amplification is claimed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Tuple

import numpy as np
import torch


def _flat_norm(update: Mapping[str, torch.Tensor]) -> float:
    if not update:
        return 0.0
    return float(torch.sqrt(sum((v.detach().float() ** 2).sum() for v in update.values())))


def clip_update(update: Mapping[str, torch.Tensor], max_norm: float
                ) -> Tuple[Dict[str, torch.Tensor], float, float]:
    """Clip the flattened update to an L2 ball of radius `max_norm`.

    Returns (clipped, original_norm, factor). Safe on an empty update (v2
    crashed here on local-only rounds).
    """
    total = _flat_norm(update)
    factor = min(1.0, max_norm / max(total, 1e-12))
    return {k: v * factor for k, v in update.items()}, total, factor


def private_aggregate(global_state: Mapping[str, torch.Tensor],
                      client_states: List[Mapping[str, torch.Tensor]],
                      keys: List[str], max_norm: float, sigma: float,
                      generator: torch.Generator | None = None
                      ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    """Clip each client's delta, sum, add N(0, sigma^2 C^2), divide by |S|."""
    n = len(client_states)
    out = {k: v.clone() for k, v in global_state.items()}
    if n == 0 or not keys:
        return out, {"dp_clip_fraction": 0.0, "dp_mean_norm": 0.0}
    acc = {k: torch.zeros_like(global_state[k], dtype=torch.float32) for k in keys}
    clipped, norms = 0, []
    for cs in client_states:
        delta = {k: cs[k].float() - global_state[k].float() for k in keys if k in cs}
        delta, norm, factor = clip_update(delta, max_norm)
        norms.append(norm); clipped += int(factor < 1.0)
        for k, v in delta.items():
            acc[k] += v
    std = sigma * max_norm
    for k in keys:
        if std > 0:
            acc[k] += torch.normal(0.0, std, size=acc[k].shape, generator=generator,
                                   device=acc[k].device, dtype=acc[k].dtype)
        out[k] = (global_state[k].float() + acc[k] / n).to(global_state[k].dtype)
    return out, {"dp_clip_fraction": clipped / n, "dp_mean_norm": float(np.mean(norms))}


# --------------------------------------------------------------------------- #
# accountant
# --------------------------------------------------------------------------- #
def rdp_gaussian(sigma: float, steps: int, orders=None) -> Tuple[np.ndarray, np.ndarray]:
    """Exact RDP of the Gaussian mechanism with noise multiplier `sigma`:
    epsilon(alpha) = alpha / (2 sigma^2) per application (Mironov, 2017,
    Prop. 7), composed additively over `steps` applications."""
    if orders is None:
        orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 256))
    orders = np.asarray(orders, dtype=float)
    if sigma <= 0:
        return orders, np.full_like(orders, np.inf)
    return orders, steps * orders / (2.0 * sigma ** 2)


def rdp_to_dp(orders: np.ndarray, rdp: np.ndarray, delta: float = 1e-5) -> float:
    """(alpha, rho)-RDP  =>  (rho + log(1/delta)/(alpha-1), delta)-DP, minimised
    over alpha (Mironov 2017, Prop. 3)."""
    with np.errstate(over="ignore", invalid="ignore"):
        eps = rdp + np.log(1.0 / delta) / (orders - 1.0)
    eps = eps[np.isfinite(eps)]
    return float(np.min(eps)) if eps.size else float("inf")


@dataclass
class DPConfig:
    enabled: bool = False
    max_norm: float = 1.0        # C
    sigma: float = 1.0           # noise multiplier
    delta: float = 1e-5

    def epsilon(self, transmitting_rounds: int) -> float:
        if not self.enabled or self.sigma <= 0 or transmitting_rounds <= 0:
            return float("inf") if self.enabled else float("inf")
        o, r = rdp_gaussian(self.sigma, transmitting_rounds)
        return rdp_to_dp(o, r, self.delta)

    def summary(self, transmitting_rounds: int) -> Dict[str, float]:
        return {"dp_enabled": float(self.enabled), "dp_sigma": self.sigma,
                "dp_max_norm": self.max_norm, "dp_delta": self.delta,
                "dp_rounds_charged": float(transmitting_rounds),
                "dp_epsilon": self.epsilon(transmitting_rounds)}

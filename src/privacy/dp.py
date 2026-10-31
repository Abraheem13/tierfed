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
Mironov (2017). v4 adds Poisson client sampling with rate q: the
subsampled-Gaussian RDP of Mironov, Talwar and Zhang (2019) is used, the sum is
divided by the fixed expected cohort size qN, and a planned release with an
empty cohort still adds noise. With q = 1 the accountant is unchanged.
"""
from __future__ import annotations

import math
from functools import lru_cache
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
                      generator: torch.Generator | None = None,
                      denominator: float | None = None
                      ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    """Clip each client's delta, sum, add N(0, sigma^2 C^2), divide by a fixed
    denominator. Under Poisson client sampling the denominator must not depend
    on the realised cohort (McMahan et al. 2018, fixed-denominator estimator),
    so the caller passes the expected cohort size q*N."""
    n = len(client_states)
    out = {k: v.clone() for k, v in global_state.items()}
    if not keys or (n == 0 and denominator is None):
        return out, {"dp_clip_fraction": 0.0, "dp_mean_norm": 0.0}
    # n == 0 with a fixed denominator (Poisson sampling drew nobody): the release
    # still happens and is pure noise, so the accounting stays exact
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
        out[k] = (global_state[k].float() + acc[k] / (denominator or n)).to(global_state[k].dtype)
    return out, {"dp_clip_fraction": clipped / max(n, 1), "dp_mean_norm": float(np.mean(norms)) if norms else 0.0}


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


def _log_add(a: float, b: float) -> float:
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m)) if m > -math.inf else -math.inf


def rdp_sampled_gaussian(q: float, sigma: float, steps: int, orders=None) -> Tuple[np.ndarray, np.ndarray]:
    """RDP of the Poisson-subsampled Gaussian mechanism at integer orders
    (Mironov, Talwar and Zhang, 2019, Sec. 3.3):
        A_alpha = sum_k C(alpha,k) (1-q)^(alpha-k) q^k exp((k^2-k)/(2 sigma^2)),
        eps(alpha) = log(A_alpha) / (alpha - 1),
    composed additively over `steps` applications. With q = 1 this equals the
    plain Gaussian mechanism alpha/(2 sigma^2)."""
    if orders is None:
        orders = list(range(2, 257))
    orders = [int(a) for a in orders]
    if sigma <= 0:
        return np.array(orders, float), np.full(len(orders), np.inf)
    if q >= 1.0:   # no subsampling: exact Gaussian RDP on the fine (fractional) order grid
        return rdp_gaussian(sigma, steps)
    return np.array(orders, float), steps * _sampled_rdp_unit(float(q), float(sigma), tuple(orders))


@lru_cache(maxsize=65536)
def _sampled_rdp_unit(q: float, sigma: float, orders: tuple) -> np.ndarray:
    from scipy.special import gammaln, logsumexp
    out = np.empty(len(orders))
    lq, l1q = math.log(q), math.log1p(-q)
    for i, a in enumerate(orders):
        k = np.arange(a + 1, dtype=float)
        terms = (gammaln(a + 1) - gammaln(k + 1) - gammaln(a - k + 1)
                 + (a - k) * l1q + k * lq + (k * k - k) / (2 * sigma ** 2))
        out[i] = logsumexp(terms) / (a - 1)
    return out


def epsilon_for(sigma: float, releases: int, q: float = 1.0, delta: float = 1e-5) -> float:
    if releases <= 0:
        return 0.0
    o, r = rdp_sampled_gaussian(q, sigma, releases)
    return rdp_to_dp(o, r, delta)


@lru_cache(maxsize=4096)
def calibrate_sigma(eps_target: float, releases: int, q: float = 1.0, delta: float = 1e-5,
                    lo: float = 0.05, hi: float = 200.0, tol: float = 1e-4) -> float:
    """Smallest noise multiplier with epsilon_for(sigma, releases, q) <= eps_target (bisection)."""
    if releases <= 0:
        return 0.0
    if epsilon_for(hi, releases, q, delta) > eps_target:
        raise ValueError("target epsilon not reachable")
    while hi - lo > tol * hi:
        mid = 0.5 * (lo + hi)
        if epsilon_for(mid, releases, q, delta) <= eps_target:
            hi = mid
        else:
            lo = mid
    return hi


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
    sample_rate: float = 1.0      # Poisson client-sampling probability q
    target_epsilon: float | None = None   # if set, sigma is calibrated to it (see server)

    def epsilon(self, transmitting_rounds: int) -> float:
        if not self.enabled or self.sigma <= 0 or transmitting_rounds <= 0:
            return float("inf")
        return epsilon_for(self.sigma, transmitting_rounds, self.sample_rate, self.delta)

    def summary(self, transmitting_rounds: int) -> Dict[str, float]:
        return {"dp_enabled": float(self.enabled), "dp_sigma": self.sigma,
                "dp_sample_rate": self.sample_rate,
                "dp_target_epsilon": float(self.target_epsilon) if self.target_epsilon else float("nan"),
                "dp_max_norm": self.max_norm, "dp_delta": self.delta,
                "dp_rounds_charged": float(transmitting_rounds),
                "dp_epsilon": self.epsilon(transmitting_rounds)}

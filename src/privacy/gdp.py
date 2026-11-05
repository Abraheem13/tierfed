"""Exact privacy of the full-participation (q = 1) Gaussian mechanism.

T releases of a Gaussian mechanism with sensitivity C and noise sigma*C compose to
mu-Gaussian DP with mu = sqrt(T)/sigma (Dong, Roth and Su), whose exact (eps, delta)
curve is delta(eps) = Phi(-eps/mu + mu/2) - e^eps Phi(-eps/mu - mu/2) (Balle and Wang).
The RDP accountant of the paper is an upper bound on this; at q = 1 the gap is the
whole looseness of the accounting, so it measures how much noise RDP over-calibrates.
"""
from __future__ import annotations

import math

from scipy.stats import norm


def delta_of_eps(eps: float, mu: float) -> float:
    return float(norm.cdf(-eps / mu + mu / 2) - math.exp(eps) * norm.cdf(-eps / mu - mu / 2))


def eps_gdp(releases: int, sigma: float, delta: float = 1e-5) -> float:
    """Smallest eps with delta(eps) <= delta for `releases` Gaussian releases at q = 1."""
    if releases <= 0:
        return 0.0
    mu = math.sqrt(releases) / sigma
    lo, hi = 0.0, 1.0
    while delta_of_eps(hi, mu) > delta:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if delta_of_eps(mid, mu) > delta:
            lo = mid
        else:
            hi = mid
    return hi


def sigma_gdp(target_eps: float, releases: int, delta: float = 1e-5) -> float:
    """Smallest sigma for which `releases` releases at q = 1 are (target_eps, delta)-DP."""
    lo, hi = 1e-3, 1.0
    while eps_gdp(releases, hi, delta) > target_eps:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if eps_gdp(releases, mid, delta) > target_eps:
            lo = mid
        else:
            hi = mid
    return hi

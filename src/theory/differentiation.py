"""Closed form of the per-tier interval programme and the value of differentiation.

Relax the intervals of  min_K sum_g K_g^2 G_g  s.t.  sum_g w_g / K_g <= rho  (w_g > 0,
sum_g w_g = 1, G_g > 0) to the positive reals. The objective is strictly convex and
increasing in every K_g, the feasible set is convex, so the quota binds and the KKT
point is the unique minimiser:

    K_g* = (w_g / G_g)^(1/3) * S / rho,    S = sum_h w_h^(2/3) G_h^(1/3),
    J*   = S^3 / rho^2.

A uniform interval K = 1/rho costs J_u = sum_g G_g / rho^2, and by Hoelder's inequality
(exponents 3/2 and 3) S^3 <= (sum_g w_g)^2 sum_g G_g = sum_g G_g, with equality iff
w_g / G_g is the same for every tier. Hence

    psi = 1 - S^3 / sum_g G_g  in [0, 1)

bounds the relative reduction of the objective that ANY per-tier schedule can achieve
over the uniform one at the same quota (for integer intervals with 1/rho an integer
the relaxation lower-bounds the integer optimum and K = 1/rho is integer-feasible).
psi is scale-free in G, so a common bias of the dispersion proxy cancels.
"""
from __future__ import annotations

from typing import Dict, Mapping

import numpy as np


def closed_form(w: Mapping[str, float], gamma: Mapping[str, float], rho: float) -> Dict[str, object]:
    tiers = sorted(w)
    W = np.array([w[g] for g in tiers], float); W = W / W.sum()
    G = np.array([gamma[g] for g in tiers], float)
    if np.any(G <= 0) or rho <= 0:
        raise ValueError("dispersions and the quota must be positive")
    S = float(np.sum(W ** (2 / 3) * G ** (1 / 3)))
    K = (W / G) ** (1 / 3) * S / rho
    J, Ju = S ** 3 / rho ** 2, float(G.sum()) / rho ** 2
    return {"K": dict(zip(tiers, K.tolist())), "J": J, "J_uniform": Ju, "psi": 1.0 - J / Ju}


def psi(w: Mapping[str, float], gamma: Mapping[str, float]) -> float:
    """Differentiation potential: the largest relative reduction of the drift
    surrogate a per-tier schedule can achieve over the uniform period."""
    return float(closed_form(w, gamma, 1.0)["psi"])

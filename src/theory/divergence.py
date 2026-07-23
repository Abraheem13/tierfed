"""Empirical convergence diagnostics for multi-rate aggregation.

Reviewers objected that NFL is a heuristic with no convergence analysis. The
companion theory (see docs/convergence.md) gives a local-SGD style bound in
which the extra error from synchronising tier g every K_g rounds enters through
the product K_g^2 * Gamma_g, where Gamma_g is the *tier-restricted* gradient
divergence. That bound is only useful if Gamma_g is small for the tiers given
long periods -- which is an empirical claim.

This module measures exactly those quantities every round:

  Gamma_g   tier-restricted gradient/parameter divergence across clients
  Delta_g   client-to-mean parameter dispersion per tier
  the bound-relevant product K_g^2 * Gamma_g

so the paper can show the assumption holds on real data instead of asserting it.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Mapping

import numpy as np
import torch


class DivergenceTracker:
    def __init__(self, tiers: Mapping[str, str] | None = None):
        self.tiers = dict(tiers) if tiers else {}
        self.rows: List[Dict[str, float]] = []

    def set_tiers(self, tiers: Mapping[str, str]):
        self.tiers = dict(tiers)

    def update(self, r: int, global_state: Mapping[str, torch.Tensor],
               client_states: List[Mapping[str, torch.Tensor]], keys: List[str]):
        if len(client_states) < 2:
            return
        per_tier_num: Dict[str, float] = defaultdict(float)
        per_tier_den: Dict[str, float] = defaultdict(float)
        tot_num = tot_den = 0.0
        for k in keys:
            vals = [cs[k].float() for cs in client_states if k in cs]
            if len(vals) < 2:
                continue
            stack = torch.stack(vals)
            mean = stack.mean(0)
            disp = float(((stack - mean) ** 2).sum(dim=tuple(range(1, stack.dim()))).mean())
            scale = float((mean ** 2).sum()) + 1e-12
            t = self.tiers.get(k, "all")
            per_tier_num[t] += disp; per_tier_den[t] += scale
            tot_num += disp; tot_den += scale
        row = {"round": float(r),
               "divergence_total": float(tot_num / max(tot_den, 1e-12)),
               "dispersion_abs": float(tot_num)}
        for t in per_tier_num:
            row[f"divergence_{t}"] = float(per_tier_num[t] / max(per_tier_den[t], 1e-12))
        self.rows.append(row)

    def latest(self) -> Dict[str, float]:
        return dict(self.rows[-1]) if self.rows else {}

    def summary(self) -> Dict[str, float]:
        if not self.rows:
            return {}
        keys = set().union(*[set(r) for r in self.rows]) - {"round"}
        out = {}
        for k in keys:
            v = np.array([r.get(k, np.nan) for r in self.rows], float)
            v = v[~np.isnan(v)]
            if v.size:
                out[f"{k}_mean"] = float(v.mean())
                out[f"{k}_final"] = float(v[-1])
        return out


def bound_terms(divergence_by_tier: Mapping[str, float], periods: Mapping[str, int],
                lr: float, rounds: int) -> Dict[str, float]:
    """Bound-relevant quantities for the multi-rate local-SGD analysis.

    For tier g synchronised every K_g rounds, the residual error contributed by
    that tier scales with lr^2 * K_g^2 * Gamma_g. Reporting the per-tier product
    shows which tier dominates and justifies giving low-divergence tiers long
    periods -- the formal counterpart of the empirical schedule.
    """
    out, total = {}, 0.0
    for g, gamma in divergence_by_tier.items():
        K = float(periods.get(g, 1))
        term = (lr ** 2) * (K ** 2) * float(gamma)
        out[f"bound_term_{g}"] = term
        total += term
    out["bound_total"] = total
    out["bound_total_x_rounds"] = total * rounds
    return out

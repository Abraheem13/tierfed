"""TierFed: adaptive multi-rate aggregation with a private head.

Design is driven by the Stage-1 diagnosis on Diabetes:
  * FedPer (private head, full-rate backbone) was the strongest baseline
    -> the personalisation head works; keep it.
  * static NFL lost ~0.05 AUROC -> a FIXED sparse schedule starves shared
    tiers precisely when their cross-client divergence Gamma_g is large
    (early training), which is exactly the K_g^2 * Gamma_g penalty in the
    convergence bound (docs/convergence.md).

TierFed therefore makes the schedule feedback-controlled by Gamma_g itself:

  1. WARMUP: for the first `warmup` rounds every shared tier syncs every
     round (identical to FedPer), so early convergence is never sacrificed.
  2. BUDGET-OPTIMAL INTERVALS: given a payload target rho, the server solves
       min sum_g K_g^2 * Gamma_g   s.t.   sum_g w_g / K_g <= rho
     greedily each round from FRESH divergence measurements, so sparsity is
     placed on the tiers where the bound says it is cheapest.
  3. LOCAL-ONLY ROUNDS: rounds where no tier is due transmit NOTHING; that
     is the mechanism by which the budget is actually met.
  4. PRIVATE HEAD: the classifier head is never transmitted (FedPer-style),
     retaining the personalisation and unlearning properties.

The claim this enables is honest and testable: accuracy non-inferior to
FedPer (margin 0.01 AUROC, one-sided test) at substantially lower upload.
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Sequence

import torch

from .base import Strategy


class AdaptiveTierFed(Strategy):
    name = "tierfed"

    def __init__(self, param_names: Sequence[str], sizes: Mapping[str, int],
                 tiers: Mapping[str, str], warmup: int = 5,
                 k_min: int = 1, k_max: int = 8,
                 rho: float = 0.5,
                 private_head: bool = True, **kw):
        super().__init__(param_names, sizes, **kw)
        self.tiers = dict(tiers)
        self.warmup = int(warmup)
        self.k_min, self.k_max = int(k_min), int(k_max)
        self.rho = float(rho)
        self.private = ([k for k, t in self.tiers.items() if t == "fast"]
                        if private_head else [])
        shared = sorted({t for t in self.tiers.values() if t != "fast"} or {"medium"})
        self.shared_tiers = shared
        self.K: Dict[str, int] = {g: self.k_min for g in shared}   # start dense
        self.last_sync: Dict[str, int] = {g: 0 for g in shared}
        self.gamma: Dict[str, float] = {}

    # ------------------------------------------------------------------ #
    def active_tiers(self, r: int) -> List[str]:
        if r <= self.warmup:
            return list(self.shared_tiers)
        # Rounds where no tier is due are pure LOCAL rounds (zero payload) --
        # that is the mechanism by which the budget rho is actually met.
        return [g for g in self.shared_tiers if (r - self.last_sync[g]) >= self.K[g]]

    def send_keys(self, r: int) -> List[str]:
        act = set(self.active_tiers(r))
        priv = set(self.private)
        return [k for k in self.param_names
                if self.tiers.get(k, "medium") in act and k not in priv]

    # ------------------------------------------------------------------ #
    def _measure_gamma(self, client_states, keys) -> Dict[str, float]:
        """Relative client dispersion per tier over the transmitted keys."""
        num: Dict[str, float] = {}
        den: Dict[str, float] = {}
        for k in keys:
            vals = [cs[k].float() for cs in client_states if k in cs]
            if len(vals) < 2:
                continue
            stack = torch.stack(vals)
            mean = stack.mean(0)
            d = float(((stack - mean) ** 2).sum(dim=tuple(range(1, stack.dim()))).mean())
            s = float((mean ** 2).sum()) + 1e-12
            g = self.tiers.get(k, "medium")
            num[g] = num.get(g, 0.0) + d
            den[g] = den.get(g, 0.0) + s
        return {g: num[g] / den[g] for g in num}

    def aggregate(self, global_state, client_states, weights, r):
        keys = self.send_keys(r)
        act = set(self.active_tiers(r))
        gam = self._measure_gamma(client_states, keys)
        self.gamma.update(gam)
        # Budget-optimal interval control: choose K to minimise the bound cost
        # sum_g K_g^2 * Gamma_g subject to mean payload fraction <= rho.
        if r >= self.warmup and self.gamma:
            self._optimise_intervals()
        for g in act:
            self.last_sync[g] = r
        new = self._weighted_average(global_state, client_states, weights, keys)
        return self.apply_dp(global_state, new, keys, len(client_states))

    def _tier_weight(self) -> Dict[str, float]:
        """Share of the SHARED payload carried by each tier."""
        priv = set(self.private)
        tot = sum(v for k, v in self.sizes.items() if k not in priv) or 1
        w: Dict[str, float] = {g: 0.0 for g in self.shared_tiers}
        for k, v in self.sizes.items():
            if k in priv:
                continue
            g = self.tiers.get(k, "medium")
            if g in w:
                w[g] += v / tot
        return w

    def _optimise_intervals(self):
        """Greedy solve: min sum_g K_g^2 Gamma_g  s.t.  sum_g w_g/K_g <= rho.

        Starting from K=1 (payload 1.0), repeatedly double the interval of the
        tier whose doubling costs the least additional bound penalty per unit
        of bandwidth saved, until the payload target is met or all tiers are
        at k_max. Fresh Gamma estimates are used each time, so intervals grow
        naturally as training converges and divergence decays.
        """
        w = self._tier_weight()
        gam = {g: max(self.gamma.get(g, 0.0), 1e-12) for g in self.shared_tiers}
        K = {g: self.k_min for g in self.shared_tiers}
        payload = lambda: sum(w[g] / K[g] for g in self.shared_tiers)  # noqa: E731
        while payload() > self.rho:
            best, best_ratio = None, None
            for g in self.shared_tiers:
                if K[g] * 2 > self.k_max:
                    continue
                dcost = gam[g] * ((K[g] * 2) ** 2 - K[g] ** 2)
                dsave = w[g] / K[g] - w[g] / (K[g] * 2)
                ratio = dcost / max(dsave, 1e-12)
                if best_ratio is None or ratio < best_ratio:
                    best, best_ratio = g, ratio
            if best is None:
                break
            K[best] *= 2
        self.K = K

    # ------------------------------------------------------------------ #
    def schedule_table(self, rounds: int) -> Dict[str, float]:
        return {"warmup": float(self.warmup), "k_min": float(self.k_min),
                "k_max": float(self.k_max), "rho": float(self.rho),
                **{f"K_{g}": float(v) for g, v in self.K.items()},
                **{f"gamma_{g}": float(v) for g, v in self.gamma.items()}}

    def describe(self):
        return {**super().describe(), "warmup": self.warmup,
                "k_min": self.k_min, "k_max": self.k_max,
                "final_intervals": dict(self.K)}

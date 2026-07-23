"""Nested Federated Learning.

Rebuilt so that the two mechanisms are INDEPENDENTLY switchable:

    private_head=True/False   -- is the classifier head withheld from aggregation?
    schedule=True/False       -- are shared tiers synchronised on multi-rate periods?

This yields a clean 2x2 factorial that attributes the effect to the right cause,
which is exactly the ablation reviewers said was missing:

    (F,F) = FedAvg          (F,T) = schedule only  (NFL-Sched)
    (T,F) = FedPer          (T,T) = full NFL

Tier membership comes from `models.tiering.assign_tiers`, so the scheme applies
to MLPs, ResNets and Transformers alike rather than a hard-coded 2-layer MLP.
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Sequence

import torch

from .base import Strategy


class NestedFL(Strategy):
    name = "nested"

    def __init__(self, param_names: Sequence[str], sizes: Mapping[str, int],
                 tiers: Mapping[str, str], k_slow: int = 5, k_med: int = 2,
                 private_head: bool = True, schedule: bool = True,
                 fallback_tier: str = "medium", **kw):
        super().__init__(param_names, sizes, **kw)
        self.tiers = dict(tiers)
        self.k = {"slow": int(k_slow), "medium": int(k_med), "fast": 1}
        self.private_head = private_head
        self.use_schedule = schedule
        self.fallback_tier = fallback_tier
        self.private = [k for k, t in self.tiers.items() if t == "fast"] if private_head else []

    # ------------------------------------------------------------------ #
    def active_tiers(self, r: int) -> List[str]:
        shared = ["slow", "medium"] if self.private_head else ["slow", "medium", "fast"]
        if not self.use_schedule:
            return shared
        act = [g for g in shared if r == 1 or r % self.k.get(g, 1) == 0]
        if not act:
            act = [self.fallback_tier] if self.fallback_tier in shared else [shared[-1]]
        return act

    def send_keys(self, r: int) -> List[str]:
        act = set(self.active_tiers(r))
        priv = set(self.private)
        return [k for k in self.param_names if self.tiers.get(k, "medium") in act and k not in priv]

    def aggregate(self, global_state, client_states, weights, r):
        keys = self.send_keys(r)
        new = self._weighted_average(global_state, client_states, weights, keys)
        return self.apply_dp(global_state, new, keys, len(client_states))

    # ------------------------------------------------------------------ #
    def payload_fraction(self, r: int) -> float:
        total = sum(self.sizes.values()) or 1
        return sum(self.sizes[k] for k in self.send_keys(r)) / total

    def schedule_table(self, rounds: int) -> Dict[str, float]:
        """Deterministic, auditable communication profile of the schedule."""
        fr = [self.payload_fraction(r) for r in range(1, rounds + 1)]
        return {"mean_payload_fraction": float(sum(fr) / len(fr)),
                "rounds_full_sync": float(sum(1 for f in fr if f > 0.99)),
                "k_slow": float(self.k["slow"]), "k_med": float(self.k["medium"]),
                "private_head": float(self.private_head), "schedule": float(self.use_schedule)}

    def describe(self):
        return {**super().describe(), "k_slow": self.k["slow"], "k_med": self.k["medium"],
                "private_head": self.private_head, "use_schedule": self.use_schedule}


def build_strategy(name: str, param_names, sizes, tiers=None, head_keys=None, **kw) -> Strategy:
    """Factory used by every experiment script."""
    from .baselines import SCAFFOLD, FedAvg, FedLAMA, FedPer, FedProx
    name = name.lower()
    common = {k: kw[k] for k in ("compressor", "dp") if k in kw}
    if name == "fedavg":
        return FedAvg(param_names, sizes, **common)
    if name == "fedprox":
        return FedProx(param_names, sizes, mu=kw.get("mu", 0.01), **common)
    if name == "fedper":
        return FedPer(param_names, sizes, head_keys=head_keys or [], **common)
    if name == "fedlama":
        return FedLAMA(param_names, sizes, tau=kw.get("tau", 5), phi=kw.get("phi", 2),
                       max_interval=kw.get("max_interval", 8), **common)
    if name == "scaffold":
        return SCAFFOLD(param_names, sizes, **common)
    if name in ("nested", "nfl", "nested_sched", "nested_nohead"):
        return NestedFL(param_names, sizes, tiers or {},
                        k_slow=kw.get("k_slow", 5), k_med=kw.get("k_med", 2),
                        private_head=kw.get("private_head", name != "nested_sched"),
                        schedule=kw.get("schedule", name != "nested_nohead"), **common)
    raise ValueError(f"unknown strategy '{name}'")

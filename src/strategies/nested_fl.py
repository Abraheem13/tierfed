"""Fixed multi-rate schedule (the negative result) and the strategy factory.

The two mechanisms remain independently switchable so the 2x2 factorial
(private head x multi-rate schedule) can be run:

    (F,F) = FedAvg      (F,T) = schedule only
    (T,F) = FedPer      (T,T) = fixed multi-rate + private head
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Sequence

from .base import Strategy


class FixedSchedule(Strategy):
    name = "fixed"

    def __init__(self, param_names: Sequence[str], sizes: Mapping[str, int],
                 tiers: Mapping[str, str], k_slow: int = 5, k_med: int = 2, warmup: int = 0,
                 private_head: bool = True, schedule: bool = True,
                 fallback_tier: str | None = None, **kw):
        super().__init__(param_names, sizes, **kw)
        self.tiers = dict(tiers)
        self.k = {"slow": int(k_slow), "medium": int(k_med), "fast": 1}
        self.warmup = int(warmup)
        self.private_head = private_head
        self.use_schedule = schedule
        self.fallback_tier = fallback_tier
        self.private = [k for k, t in self.tiers.items() if t == "fast"] if private_head else []

    def active_tiers(self, r: int) -> List[str]:
        shared = ["slow", "medium"] if self.private_head else ["slow", "medium", "fast"]
        if not self.use_schedule or r <= self.warmup or r == 1:
            return shared
        act = [g for g in shared if r % self.k.get(g, 1) == 0]
        # v3: rounds in which no tier is due are local-only (zero payload), exactly
        # as in TierFed. An earlier version sent a fallback tier in such rounds,
        # which made the medium-tier period ineffective; it is kept only as an
        # explicit opt-in (fallback_tier="medium") for reproducing that version.
        if not act and self.fallback_tier in shared:
            act = [self.fallback_tier]
        return act

    def send_keys(self, r: int) -> List[str]:
        act = set(self.active_tiers(r)); priv = set(self.private)
        return [k for k in self.param_names if self.tiers.get(k, "medium") in act and k not in priv]

    def payload_fraction(self, r: int) -> float:
        total = sum(self.sizes.values()) or 1
        return sum(self.sizes[k] for k in self.send_keys(r)) / total

    def schedule_table(self, rounds: int) -> Dict[str, float]:
        fr = [self.payload_fraction(r) for r in range(1, rounds + 1)]
        return {"mean_payload_fraction": float(sum(fr) / len(fr)),
                "k_slow": float(self.k["slow"]), "k_med": float(self.k["medium"]),
                "private_head": float(self.private_head), "schedule": float(self.use_schedule)}

    def describe(self):
        return {**super().describe(), "k_slow": self.k["slow"], "k_med": self.k["medium"],
                "private_head": self.private_head, "use_schedule": self.use_schedule}


# backwards-compatible alias
NestedFL = FixedSchedule


def build_strategy(name: str, param_names, sizes, tiers=None, head_keys=None, **kw) -> Strategy:
    """Factory used by every experiment script."""
    from .baselines import LAG, SCAFFOLD, AdaComm, FedAvg, FedLAMA, FedPer, FedProx
    from .adaptive import AdaptiveTierFed
    name = name.lower()
    common = {k: kw[k] for k in ("compressor",) if k in kw}
    if name == "fedavg":
        return FedAvg(param_names, sizes, **common)
    if name == "fedprox":
        return FedProx(param_names, sizes, mu=kw.get("mu", 0.01), **common)
    if name == "fedper":
        return FedPer(param_names, sizes, head_keys=head_keys or [], **common)
    if name == "fedlama":
        return FedLAMA(param_names, sizes, tau=kw.get("tau", 5), phi=kw.get("phi", 2),
                       base_interval=kw.get("base_interval", 1),
                       max_interval=kw.get("max_interval", 8), **common)
    if name == "scaffold":
        return SCAFFOLD(param_names, sizes, **common)
    if name == "adacomm":
        return AdaComm(param_names, sizes, tau0=kw.get("tau0", 8),
                       max_interval=kw.get("max_interval", 8), **common)
    if name == "lag":
        return LAG(param_names, sizes, xi=kw.get("xi", 1.0), **common)
    if name == "tierfed":
        return AdaptiveTierFed(param_names, sizes, tiers or {},
                               warmup=kw.get("warmup", 5),
                               k_min=kw.get("k_min", 1), k_max=kw.get("k_max", 8),
                               rho=kw.get("rho", 0.25), solver=kw.get("solver", "exact"),
                               private_head=kw.get("private_head", True), **common)
    if name in ("fixed", "nested", "nfl", "nested_sched", "nested_nohead"):
        return FixedSchedule(param_names, sizes, tiers or {},
                             k_slow=kw.get("k_slow", 5), k_med=kw.get("k_med", 2),
                             warmup=kw.get("warmup", 0),
                             private_head=kw.get("private_head", name != "nested_sched"),
                             schedule=kw.get("schedule", name != "nested_nohead"), **common)
    raise ValueError(f"unknown strategy '{name}'")

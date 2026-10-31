"""TierFed: budget-constrained, divergence-driven multi-rate aggregation.

Each round the server (i) measures the relative cross-client dispersion
Gamma_hat_g of every tier it received, (ii) solves

    min_{K_g in [1, K_max]}  sum_g K_g^2 Gamma_hat_g   s.t.  sum_g w_g / K_g <= rho

and (iii) synchronises tier g only when r - tau_g >= K_g. Rounds in which no
tier is due transmit nothing.

v3 changes relative to v2:
  * the programme is solved EXACTLY by enumeration (|G| <= 3 shared tiers and
    K_max = 8 give at most 512 points), so "budget-optimal" is literally true
    for the surrogate; the greedy doubling rule of v2 is kept as `solver=
    "greedy"` for models with many tiers and is restricted to powers of two,
    which the paper now states;
  * the realised payload fraction (including warmup and the private head) is
    recorded, since rho is a quota on the shared payload after warmup;
  * Gamma_hat for tiers not received this round is carried forward and the
    age of each estimate is logged, so the paper can show how stale the
    controller's inputs are.
"""
from __future__ import annotations

import itertools
from typing import Dict, List, Mapping, Sequence

import torch

from .base import Strategy


class AdaptiveTierFed(Strategy):
    name = "tierfed"

    def __init__(self, param_names: Sequence[str], sizes: Mapping[str, int],
                 tiers: Mapping[str, str], warmup: int = 5,
                 k_min: int = 1, k_max: int = 8, rho: float = 0.25,
                 solver: str = "exact", private_head: bool = True, **kw):
        super().__init__(param_names, sizes, **kw)
        self.tiers = dict(tiers)
        self.warmup = int(warmup)
        self.k_min, self.k_max = int(k_min), int(k_max)
        self.rho = float(rho)
        self.solver = solver
        self.private = ([k for k, t in self.tiers.items() if t == "fast"]
                        if private_head else [])
        shared = sorted({t for t in self.tiers.values() if t != "fast"} or {"medium"})
        if not private_head and "fast" in set(self.tiers.values()):
            shared = sorted(set(shared) | {"fast"})
        self.shared_tiers = shared
        self.K: Dict[str, int] = {g: self.k_min for g in shared}
        self.last_sync: Dict[str, int] = {g: 0 for g in shared}
        self.gamma: Dict[str, float] = {}
        self.gamma_age: Dict[str, int] = {}
        self.K_history: List[Dict[str, int]] = []

    # ------------------------------------------------------------------ #
    def active_tiers(self, r: int) -> List[str]:
        if r <= self.warmup:
            return list(self.shared_tiers)
        return [g for g in self.shared_tiers if (r - self.last_sync[g]) >= self.K[g]]

    def send_keys(self, r: int) -> List[str]:
        act = set(self.active_tiers(r)); priv = set(self.private)
        return [k for k in self.param_names
                if self.tiers.get(k, "medium") in act and k not in priv]

    # ------------------------------------------------------------------ #
    def _measure_gamma(self, client_states, keys) -> Dict[str, float]:
        """Relative client dispersion per tier over the transmitted keys:
        sum_j mean_k ||theta_kj - mean_j||^2 / (sum_j ||mean_j||^2 + eps)."""
        num: Dict[str, float] = {}; den: Dict[str, float] = {}
        for k in keys:
            vals = [cs[k].float() for cs in client_states if k in cs]
            if len(vals) < 2:
                continue
            stack = torch.stack(vals); mean = stack.mean(0)
            d = float(((stack - mean) ** 2).sum(dim=tuple(range(1, stack.dim()))).mean())
            s = float((mean ** 2).sum())
            g = self.tiers.get(k, "medium")
            num[g] = num.get(g, 0.0) + d; den[g] = den.get(g, 0.0) + s
        return {g: num[g] / (den[g] + 1e-12) for g in num}

    def observe(self, global_state, client_states, keys, r):
        act = set(self.active_tiers(r))
        gam = self._measure_gamma(client_states, keys)
        for g in self.shared_tiers:
            if g in gam:
                self.gamma[g] = gam[g]; self.gamma_age[g] = 0
            elif g in self.gamma:
                self.gamma_age[g] = self.gamma_age.get(g, 0) + 1
        for g in act:
            self.last_sync[g] = r
        if r >= self.warmup and self.gamma:
            self.K = self._solve()
        self.K_history.append(dict(self.K))

    def aggregate(self, global_state, client_states, weights, r):
        keys = self.send_keys(r)
        return self._weighted_average(global_state, client_states, weights, keys)

    # ------------------------------------------------------------------ #
    def tier_weight(self) -> Dict[str, float]:
        """Share of the SHARED payload carried by each tier (w_g)."""
        priv = set(self.private)
        tot = sum(v for k, v in self.sizes.items() if k not in priv) or 1
        w = {g: 0.0 for g in self.shared_tiers}
        for k, v in self.sizes.items():
            if k in priv:
                continue
            g = self.tiers.get(k, "medium")
            if g in w:
                w[g] += v / tot
        return w

    def _solve(self) -> Dict[str, int]:
        w = self.tier_weight()
        gam = {g: max(self.gamma.get(g, 0.0), 1e-12) for g in self.shared_tiers}
        if self.solver == "greedy":
            return self._solve_greedy(w, gam)
        return self._solve_exact(w, gam)

    def _solve_exact(self, w, gam) -> Dict[str, int]:
        tiers = list(self.shared_tiers)
        grid = range(self.k_min, self.k_max + 1)
        best, best_cost = None, None
        for Ks in itertools.product(grid, repeat=len(tiers)):
            payload = sum(w[g] / K for g, K in zip(tiers, Ks))
            feasible = payload <= self.rho + 1e-12
            cost = sum(gam[g] * K * K for g, K in zip(tiers, Ks))
            # lexicographic: feasible first, then bound cost, then smaller K (denser)
            key = (0 if feasible else 1, cost if feasible else payload, sum(Ks))
            if best_cost is None or key < best_cost:
                best, best_cost = dict(zip(tiers, Ks)), key
        return best

    def _solve_greedy(self, w, gam) -> Dict[str, int]:
        """v2 rule: double the interval of the tier with the least bound cost per
        unit bandwidth saved; restricted to powers of two."""
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
        return K

    # ------------------------------------------------------------------ #
    def schedule_table(self, rounds: int) -> Dict[str, float]:
        out = {"warmup": float(self.warmup), "k_min": float(self.k_min),
               "k_max": float(self.k_max), "rho": float(self.rho),
               **{f"K_{g}": float(v) for g, v in self.K.items()},
               **{f"gamma_{g}": float(v) for g, v in self.gamma.items()},
               **{f"w_{g}": float(v) for g, v in self.tier_weight().items()}}
        if self.K_history:
            for g in self.shared_tiers:
                ks = [h[g] for h in self.K_history if g in h]
                out[f"K_{g}_mean"] = float(sum(ks) / len(ks))
            distinct = len({tuple(sorted(h.items())) for h in self.K_history})
            out["n_distinct_schedules"] = float(distinct)
            differentiated = sum(1 for h in self.K_history if len(set(h.values())) > 1)
            out["frac_rounds_tier_differentiated"] = float(differentiated / len(self.K_history))
        return out

    def describe(self):
        return {**super().describe(), "warmup": self.warmup, "solver": self.solver,
                "k_min": self.k_min, "k_max": self.k_max, "rho": self.rho,
                "final_intervals": dict(self.K)}


class PrivacyPlannedTierFed(Strategy):
    """TierFed-DP: a privacy- and byte-budgeted schedule planned BEFORE training.

    Under client-level DP the set of transmitting rounds is observable, so a
    schedule computed from client updates (adaptive TierFed, AdaComm, LAG,
    FedLAMA) is itself a data-dependent release that the accountant does not
    cover. TierFed-DP therefore plans offline, from public quantities only:
    the privacy target (epsilon, delta), the client-sampling rate q, the number
    of rounds R, the warm-up T_w, the tier sizes d_g and the byte quota rho.

    For every candidate interval pair (K_s, K_m) in {1..K_max}^2 it computes the
    resulting release rounds, the noise multiplier sigma(K) that makes the run
    exactly epsilon-DP for that number of releases (exact subsampled-Gaussian
    RDP accountant), and the surrogate

        J(K) = sum_g w_g K_g^2  +  lam * (1/R) sum_r (d_tx(r)/d) * sigma(K)^2,

    i.e. the multi-rate drift term of the bound (with a flat, data-free
    heterogeneity prior) plus the noise energy injected into the shared model
    per round. Fewer releases need less noise per release (sigma^2 grows roughly
    linearly with the release count) but drift more; J trades the two. The
    pair minimising J subject to the byte quota is used, and the server is told
    sigma. lam is the single prior (it absorbs the unknown smoothness and
    heterogeneity constants of the bound); its sensitivity is reported.
    """
    name = "tierfed_dp"
    dp_safe = True

    def __init__(self, param_names, sizes, tiers, rounds: int = 60, warmup: int = 5,
                 epsilon: float = 4.0, delta: float = 1e-5, q: float = 1.0,
                 k_max: int = 8, rho: float = 1.0, lam: float = 1.0,
                 private_head: bool = True, **kw):
        super().__init__(param_names, sizes, **kw)
        from ..privacy.dp import calibrate_sigma
        self.tiers = dict(tiers)
        self.private = [k for k, t in self.tiers.items() if t == "fast"] if private_head else []
        self.shared_tiers = sorted({t for t in self.tiers.values() if t != "fast"} or {"medium"})
        self.R, self.Tw, self.k_max, self.rho, self.lam = int(rounds), int(warmup), int(k_max), float(rho), float(lam)
        self.eps, self.delta, self.q = float(epsilon), float(delta), float(q)
        priv = set(self.private)
        tot = sum(v for k, v in self.sizes.items() if k not in priv) or 1
        self.w = {g: sum(v for k, v in self.sizes.items() if self.tiers.get(k) == g and k not in priv) / tot
                  for g in self.shared_tiers}
        best = None
        self.candidates = []
        for Ks in itertools.product(range(1, self.k_max + 1), repeat=len(self.shared_tiers)):
            K = dict(zip(self.shared_tiers, Ks))
            due = [[g for g in self.shared_tiers if r <= self.Tw or (r - self.Tw) % K[g] == 0]
                   for r in range(1, self.R + 1)]
            frac = [sum(self.w[g] for g in d) for d in due]
            releases = sum(1 for f in frac if f > 0)
            payload = sum(frac) / self.R
            if payload > self.rho + 1e-12 or releases == 0:
                continue
            sigma = calibrate_sigma(self.eps, releases, self.q, self.delta)
            drift = sum(self.w[g] * K[g] ** 2 for g in self.shared_tiers)
            noise = sum(frac) / self.R * sigma ** 2
            J = drift + self.lam * noise
            self.candidates.append((J, K, sigma, releases, payload))
            if best is None or J < best[0]:
                best = (J, K, sigma, releases, payload)
        if best is None:
            raise ValueError("no schedule satisfies the byte quota")
        self.J, self.K, self.planned_sigma, self.planned_releases, self.planned_payload = best

    def due(self, r: int):
        return [g for g in self.shared_tiers if r <= self.Tw or (r - self.Tw) % self.K[g] == 0]

    def send_keys(self, r: int):
        act = set(self.due(r)); priv = set(self.private)
        return [k for k in self.param_names if self.tiers.get(k, "medium") in act and k not in priv]

    def planned_release_rounds(self, rounds: int):
        return [r for r in range(1, rounds + 1) if self.send_keys(r)]

    def schedule_table(self, rounds: int):
        return {"K_slow": float(self.K.get("slow", 0)), "K_medium": float(self.K.get("medium", 0)),
                "planned_sigma": self.planned_sigma, "planned_releases": float(self.planned_releases),
                "planned_payload": self.planned_payload, "lam": self.lam, "J": self.J,
                "target_epsilon": self.eps, "q": self.q}

    def describe(self):
        return {**super().describe(), **self.schedule_table(self.R)}

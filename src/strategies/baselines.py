"""Baseline strategies.

FedAvg, FedProx, FedPer, SCAFFOLD (real control variates), FedLAMA
(cumulative-share rule), AdaComm (adaptive period from loss) and LAG (lazily
aggregated updates). The last two were missing from v2's related work and are
the closest prior art to a divergence-driven communication schedule, so they
are implemented as baselines rather than discussed away.
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Sequence

import numpy as np
import torch

from .base import Strategy


class FedAvg(Strategy):
    """McMahan et al. 2017: sample-weighted average of every parameter each round."""
    name = "fedavg"
    dp_safe = True


class FedProx(Strategy):
    """Li et al. 2020: FedAvg plus a proximal term (mu/2)||theta - theta_global||^2."""
    name = "fedprox"
    dp_safe = True

    def __init__(self, *a, mu: float = 0.01, **kw):
        super().__init__(*a, **kw)
        self.mu = float(mu)

    def local_objective_extra(self, model, global_params, batch_loss, cid: int = -1):
        if self.mu <= 0 or not global_params:
            return batch_loss
        prox = 0.0
        for n, p in model.named_parameters():
            if n in global_params:
                prox = prox + ((p - global_params[n].to(p.device)) ** 2).sum()
        return batch_loss + 0.5 * self.mu * prox

    def describe(self):
        return {**super().describe(), "mu": self.mu}


class FedPer(Strategy):
    """Arivazhagan et al. 2019: shared backbone every round, private classifier head."""
    name = "fedper"
    dp_safe = True

    def __init__(self, param_names, sizes, head_keys: Sequence[str], **kw):
        super().__init__(param_names, sizes, **kw)
        self.private = list(head_keys)

    def describe(self):
        return {**super().describe(), "n_private": len(self.private)}


class SCAFFOLD(Strategy):
    """Karimireddy et al. 2020, option II control variates.

    Local gradient correction g + c - c_k is implemented as the linear term
    <c - c_k, theta> added to the loss. After local training with S steps at
    learning rate eta, c_k <- c_k - c + (theta_global - theta_k) / (S eta);
    the server updates c <- c + mean_k(delta c_k). Clients upload the model
    slice and delta c_k, so the payload is doubled (upload_multiplier = 2).
    Note: the local solver is Adam for every method in this benchmark, so the
    1/(S eta) rescaling uses the nominal learning rate; this is stated in the
    paper as an approximation of the SGD analysis.
    """
    name = "scaffold"
    upload_multiplier = 2.0

    def __init__(self, param_names, sizes, **kw):
        super().__init__(param_names, sizes, **kw)
        self.c_global: Dict[str, torch.Tensor] = {}
        self.c_local: Dict[int, Dict[str, torch.Tensor]] = {}
        self._pending: List[Dict[str, torch.Tensor]] = []

    def _c(self, store: Dict[str, torch.Tensor], k: str, like: torch.Tensor):
        if k not in store:
            store[k] = torch.zeros_like(like, dtype=torch.float32)
        return store[k]

    def local_objective_extra(self, model, global_params, batch_loss, cid: int = -1):
        corr = 0.0
        ck = self.c_local.setdefault(cid, {})
        for n, p in model.named_parameters():
            if n not in (global_params or {}):
                continue
            c = self._c(self.c_global, n, p).to(p.device)
            ci = self._c(ck, n, p).to(p.device)
            corr = corr + ((c - ci) * p).sum()
        return batch_loss + corr

    def after_local(self, cid, global_state, client_state, keys, n_steps, lr):
        ck = self.c_local.setdefault(cid, {})
        delta_c = {}
        scale = 1.0 / max(n_steps * lr, 1e-12)
        for k in keys:
            if k not in client_state:
                continue
            c = self._c(self.c_global, k, global_state[k])
            ci = self._c(ck, k, global_state[k])
            new_ci = ci - c + (global_state[k].float() - client_state[k].float()).cpu() * scale
            delta_c[k] = new_ci - ci
            ck[k] = new_ci
        self._pending.append(delta_c)

    def aggregate(self, global_state, client_states, weights, r):
        new = super().aggregate(global_state, client_states, weights, r)
        if self._pending:
            n_total = max(len(self.c_local), 1)
            for dc in self._pending:
                for k, v in dc.items():
                    self.c_global[k] = self._c(self.c_global, k, v) + v / n_total
            self._pending = []
        return new


class FedLAMA(Strategy):
    """Lee et al. AAAI 2023, layer-wise adaptive model aggregation.

    FedLAMA-style reimplementation from the paper's description (no reference
    code was used): every `tau` rounds the server measures the per-layer
    discrepancy d_l = mean_k ||theta_l^k - theta_l_bar|| of the layers it
    received and lengthens the interval of layers whose per-parameter
    discrepancy is below the model-wide average (phi x base interval), while
    the others keep the base interval. Intervals are capped at `max_interval`
    and everything is synchronised in round 1. The paper labels this
    "FedLAMA (reimpl.)".
    """
    name = "fedlama"

    def __init__(self, param_names, sizes, tau: int = 5, phi: int = 2,
                 base_interval: int = 1, max_interval: int = 8, **kw):
        super().__init__(param_names, sizes, **kw)
        self.tau, self.phi = int(tau), int(phi)
        self.base_interval, self.max_interval = int(base_interval), int(max_interval)
        self.intervals: Dict[str, int] = {k: self.base_interval for k in self.param_names}
        self.last_sync: Dict[str, int] = {k: 0 for k in self.param_names}
        self._disc: Dict[str, float] = {}

    def send_keys(self, r: int) -> List[str]:
        if r <= 1:
            return list(self.param_names)
        return [k for k in self.param_names if (r - self.last_sync[k]) >= self.intervals[k]]

    def observe(self, global_state, client_states, keys, r):
        for k in keys:
            vals = [cs[k].float() for cs in client_states if k in cs]
            if len(vals) < 2:
                continue
            mean = torch.stack(vals).mean(0)
            self._disc[k] = float(torch.stack([(v - mean).norm() for v in vals]).mean())
            self.last_sync[k] = r
        if r % self.tau == 0 and self._disc:
            self._reassign()

    def _reassign(self):
        """Layers whose per-parameter RMS discrepancy d_l / sqrt(|theta_l|) is
        below the model-wide parameter-weighted RMS discrepancy receive the
        long interval phi * tau_base; the others keep tau_base."""
        keys = [k for k in self.param_names if k in self._disc]
        if not keys:
            return
        d2 = np.array([self._disc[k] ** 2 for k in keys], float)
        s = np.array([self.sizes[k] for k in keys], float)
        global_rms2 = d2.sum() / max(s.sum(), 1e-12)
        for k, dd, ss in zip(keys, d2, s):
            long = (dd / max(ss, 1e-12)) < global_rms2
            self.intervals[k] = min(self.max_interval, self.base_interval * (self.phi if long else 1))

    def schedule_table(self, rounds):
        return {"mean_interval": float(np.mean(list(self.intervals.values()))),
                "n_long": float(sum(v > self.base_interval for v in self.intervals.values()))}

    def describe(self):
        return {**super().describe(), "tau": self.tau, "phi": self.phi,
                "max_interval": self.max_interval, **self.schedule_table(0)}


class AdaComm(Strategy):
    """Wang & Joshi, MLSys 2019: adaptive communication period for local SGD.

    All shared parameters share one period tau that is re-derived from the
    training loss: tau_j = ceil( sqrt(F_j / F_0) * tau_0 ), so the period
    shrinks as the loss falls (AdaComm, Eq. 20 in the paper). F_0 is the
    mean client loss of round 1. Here one period unit is one round.
    """
    name = "adacomm"

    def __init__(self, param_names, sizes, tau0: int = 8, max_interval: int = 8, **kw):
        super().__init__(param_names, sizes, **kw)
        self.tau0, self.max_interval = int(tau0), int(max_interval)
        self.tau = 1
        self.F0 = None
        self.last_sync = 0

    def send_keys(self, r: int) -> List[str]:
        if r <= 1 or (r - self.last_sync) >= self.tau:
            return list(self.param_names)
        return []

    def observe_loss(self, r: int, mean_loss: float):
        if self.F0 is None:
            self.F0 = max(mean_loss, 1e-12)
        self.tau = int(np.clip(np.ceil(np.sqrt(mean_loss / self.F0) * self.tau0), 1, self.max_interval))

    def observe(self, global_state, client_states, keys, r):
        if keys:
            self.last_sync = r

    def schedule_table(self, rounds):
        return {"tau_final": float(self.tau), "tau0": float(self.tau0)}


class LAG(Strategy):
    """Chen et al. NeurIPS 2018, Lazily Aggregated Gradient (worker-side rule, D=1).

    Client k uploads its update only if it changed enough since its last
    upload:  ||u_k^r - u_k^last||^2 > (xi / N^2) ||theta^r - theta^{r-1}||^2.
    Otherwise the server reuses the stale update. Clients that skip upload
    zero bits; the first round is always transmitted.
    """
    name = "lag"

    def __init__(self, param_names, sizes, xi: float = 1.0, **kw):
        super().__init__(param_names, sizes, **kw)
        self.xi = float(xi)
        self.stale: Dict[int, Dict[str, torch.Tensor]] = {}
        self.prev_global: Dict[str, torch.Tensor] | None = None
        self.n_clients = 1

    def client_sends(self, cid, global_state, client_state, keys, r):
        self.n_clients = max(self.n_clients, cid + 1)
        upd = {k: client_state[k].float() - global_state[k].float() for k in keys if k in client_state}
        if cid not in self.stale or self.prev_global is None:
            self.stale[cid] = upd
            return True
        last = self.stale[cid]
        diff = sum(((upd[k] - last[k]) ** 2).sum() for k in upd if k in last)
        move = sum(((global_state[k].float() - self.prev_global[k].float()) ** 2).sum() for k in keys)
        send = bool(diff > (self.xi / self.n_clients ** 2) * move)
        if send:
            self.stale[cid] = upd
        return send

    def aggregate(self, global_state, client_states, weights, r):
        """Server averages the freshest update of every client (stale or new)."""
        keys = self.send_keys(r)
        self.prev_global = {k: v.clone() for k, v in global_state.items()}
        states = [{k: global_state[k].float() + u[k] for k in u} for u in self.stale.values()]
        w = [1.0] * len(states)
        return self._weighted_average(global_state, states, w, keys)

    def schedule_table(self, rounds):
        return {"xi": self.xi}

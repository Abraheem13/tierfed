"""Strategy interface.

A Strategy answers three questions each round:
  send_keys(r)      -- which parameters the server broadcasts / expects back
  personal_keys()   -- which parameters every client keeps private forever
  aggregate(...)    -- how returned slices become the next global state

Communication is accounted exactly: uploaded bits are measured from what the
compressor actually emitted, not from a nominal parameter count.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Sequence

import torch

from ..compression import Compressor, NoCompression
from ..privacy.dp import DPConfig, add_gaussian_noise, clip_update


@dataclass
class RoundReport:
    round: int
    active_keys: List[str]
    upload_bits: float = 0.0
    download_bits: float = 0.0
    extra: Dict[str, float] = field(default_factory=dict)


class Strategy:
    name = "base"
    #: parameters never transmitted (personalisation tier)
    private: Sequence[str] = ()

    def __init__(self, param_names: Sequence[str], sizes: Mapping[str, int],
                 compressor: Compressor | None = None, dp: DPConfig | None = None):
        self.param_names = list(param_names)
        self.sizes = dict(sizes)
        self.compressor = compressor or NoCompression()
        self.dp = dp or DPConfig()
        self.history: List[RoundReport] = []

    # ---------------- schedule ----------------
    def send_keys(self, r: int) -> List[str]:
        return [k for k in self.param_names if k not in set(self.private)]

    def personal_keys(self) -> List[str]:
        return list(self.private)

    # ---------------- aggregation ----------------
    def aggregate(self, global_state: Dict[str, torch.Tensor],
                  client_states: List[Dict[str, torch.Tensor]],
                  weights: Sequence[float], r: int) -> Dict[str, torch.Tensor]:
        keys = self.send_keys(r)
        return self._weighted_average(global_state, client_states, weights, keys)

    # ---------------- helpers ----------------
    def _weighted_average(self, global_state, client_states, weights, keys):
        w = torch.tensor(list(weights), dtype=torch.float32)
        w = w / w.sum().clamp_min(1e-12)
        new = {k: v.clone() for k, v in global_state.items()}
        for k in keys:
            present = [(cs[k], wi) for cs, wi in zip(client_states, w) if k in cs]
            if not present:
                continue
            acc = torch.zeros_like(global_state[k], dtype=torch.float32)
            tw = 0.0
            for v, wi in present:
                acc += v.float() * float(wi); tw += float(wi)
            if tw > 0:
                new[k] = (acc / tw).to(global_state[k].dtype)
        return new

    def apply_dp(self, global_state, aggregated, keys, n_clients, generator=None):
        """Client-level DP applied to the aggregated update (delta form)."""
        if not self.dp.enabled:
            return aggregated
        delta = {k: aggregated[k].float() - global_state[k].float() for k in keys}
        delta, _, _ = clip_update(delta, self.dp.max_norm)
        delta = add_gaussian_noise(delta, self.dp.sigma, self.dp.max_norm, n_clients, generator)
        out = {k: v.clone() for k, v in aggregated.items()}
        for k in keys:
            out[k] = (global_state[k].float() + delta[k]).to(global_state[k].dtype)
        return out

    def measure_upload(self, client_states: List[Dict[str, torch.Tensor]], keys) -> float:
        bits = 0.0
        for cs in client_states:
            for k in keys:
                if k in cs:
                    _, b = self.compressor(cs[k])
                    bits += b
        return bits

    def compress_client_state(self, state: Dict[str, torch.Tensor], keys):
        out = dict(state)
        for k in keys:
            if k in out:
                out[k], _ = self.compressor(out[k])
        return out

    def local_objective_extra(self, model, global_params, batch_loss):
        """Hook for proximal / correction terms (FedProx, SCAFFOLD)."""
        return batch_loss

    def describe(self) -> Dict[str, object]:
        return {"strategy": self.name, "private_params": list(self.private),
                "compressor": self.compressor.name, "dp": self.dp.summary(1)}

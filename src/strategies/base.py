"""Strategy interface.

A Strategy answers, each round r:
  send_keys(r)              which parameters are broadcast and expected back
  personal_keys()           which parameters every client keeps private forever
  client_sends(cid, ...)    whether a given client uploads this round (LAG-style
                            laziness; every other strategy returns True)
  local_objective_extra     proximal / control-variate terms for local training
  after_local(...)          per-client bookkeeping after local training
  aggregate(...)            how returned slices become the next global state

Differential privacy is NOT applied here: the server applies the per-client
clip-and-noise mechanism uniformly to every strategy (see server.py), which
fixes the v2 defect in which only two strategies ever saw DP.

Communication is accounted exactly: uploaded bits are what the compressor
actually emitted for the clients that actually sent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Sequence

import torch

from ..compression import Compressor, NoCompression


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
    #: upload multiplier for auxiliary state (SCAFFOLD sends control variates)
    upload_multiplier: float = 1.0

    def __init__(self, param_names: Sequence[str], sizes: Mapping[str, int],
                 compressor: Compressor | None = None, **_):
        self.param_names = list(param_names)
        self.sizes = dict(sizes)
        self.compressor = compressor or NoCompression()
        self.history: List[RoundReport] = []

    # ---------------- schedule ----------------
    def send_keys(self, r: int) -> List[str]:
        priv = set(self.private)
        return [k for k in self.param_names if k not in priv]

    def personal_keys(self) -> List[str]:
        return list(self.private)

    def client_sends(self, cid: int, global_state, client_state, keys, r: int) -> bool:
        return True

    # ---------------- local training hooks ----------------
    def local_objective_extra(self, model, global_params, batch_loss, cid: int = -1):
        return batch_loss

    def after_local(self, cid: int, global_state, client_state, keys, n_steps: int, lr: float):
        return None

    # ---------------- aggregation ----------------
    def aggregate(self, global_state: Dict[str, torch.Tensor],
                  client_states: List[Dict[str, torch.Tensor]],
                  weights: Sequence[float], r: int) -> Dict[str, torch.Tensor]:
        keys = self.send_keys(r)
        return self._weighted_average(global_state, client_states, weights, keys)

    def observe(self, global_state, client_states, keys, r: int):
        """Called by the server after aggregation with the raw client slices
        (before DP noise), for divergence-driven schedulers."""
        return None

    # ---------------- helpers ----------------
    @staticmethod
    def _weighted_average(global_state, client_states, weights, keys):
        w = torch.tensor(list(weights), dtype=torch.float32)
        w = w / w.sum().clamp_min(1e-12)
        new = {k: v.clone() for k, v in global_state.items()}
        for k in keys:
            present = [(cs[k], float(wi)) for cs, wi in zip(client_states, w) if k in cs]
            if not present:
                continue
            acc = torch.zeros_like(global_state[k], dtype=torch.float32)
            tw = 0.0
            for v, wi in present:
                acc += v.float() * wi; tw += wi
            if tw > 0:
                new[k] = (acc / tw).to(global_state[k].dtype)
        return new

    def measure_upload(self, client_states: List[Dict[str, torch.Tensor]], keys) -> float:
        bits = 0.0
        for cs in client_states:
            for k in keys:
                if k in cs:
                    _, b = self.compressor(cs[k])
                    bits += b
        return bits * self.upload_multiplier

    def compress_client_state(self, state: Dict[str, torch.Tensor], keys):
        out = dict(state)
        for k in keys:
            if k in out:
                out[k], _ = self.compressor(out[k])
        return out

    def schedule_table(self, rounds: int) -> Dict[str, float]:
        return {}

    def describe(self) -> Dict[str, object]:
        return {"strategy": self.name, "private_params": list(self.private),
                "compressor": self.compressor.name}

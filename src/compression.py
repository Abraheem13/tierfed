"""Communication compressors.

Reviewers noted the 32.6% bandwidth claim was never benchmarked against
standard compression. These operators are (a) baselines in their own right and
(b) composable with the multi-frequency schedule, so the paper can report
NFL x compression jointly rather than claiming they are alternatives.

Every compressor returns (tensor, bits_actually_sent) so accounting is exact
rather than nominal.
"""
from __future__ import annotations

from typing import Dict, Mapping, Tuple

import numpy as np
import torch

FP32_BITS = 32


class Compressor:
    name = "none"

    def __call__(self, t: torch.Tensor) -> Tuple[torch.Tensor, float]:
        return t, t.numel() * FP32_BITS

    def compress_state(self, state: Mapping[str, torch.Tensor]):
        out, bits = {}, 0.0
        for k, v in state.items():
            q, b = self(v)
            out[k], bits = q, bits + b
        return out, bits


class NoCompression(Compressor):
    name = "none"


class UniformQuantizer(Compressor):
    """b-bit uniform scalar quantisation with per-tensor min/max scaling."""

    def __init__(self, bits: int = 8):
        self.bits = bits
        self.name = f"quant{bits}"

    def __call__(self, t):
        if t.numel() == 0:
            return t, 0.0
        lo, hi = t.min(), t.max()
        levels = 2 ** self.bits - 1
        scale = (hi - lo).clamp_min(1e-12) / levels
        q = torch.round((t - lo) / scale)
        deq = q * scale + lo
        return deq, t.numel() * self.bits + 2 * FP32_BITS   # payload + min/max


class TopKSparsifier(Compressor):
    """Top-k magnitude sparsification; sends values plus indices."""

    def __init__(self, k_ratio: float = 0.1):
        self.k = k_ratio
        self.name = f"topk{int(k_ratio*100)}"

    def __call__(self, t):
        n = t.numel()
        if n == 0:
            return t, 0.0
        k = max(1, int(n * self.k))
        flat = t.flatten()
        idx = torch.topk(flat.abs(), k).indices
        out = torch.zeros_like(flat)
        out[idx] = flat[idx]
        index_bits = k * max(1.0, np.ceil(np.log2(max(n, 2))))
        return out.view_as(t), k * FP32_BITS + index_bits


class SparseTernary(Compressor):
    """Sattler et al. sparse ternary compression: top-k support, ternary values."""

    def __init__(self, k_ratio: float = 0.05):
        self.k = k_ratio
        self.name = f"stc{int(k_ratio*100)}"

    def __call__(self, t):
        n = t.numel()
        if n == 0:
            return t, 0.0
        k = max(1, int(n * self.k))
        flat = t.flatten()
        idx = torch.topk(flat.abs(), k).indices
        mu = flat[idx].abs().mean()
        out = torch.zeros_like(flat)
        out[idx] = mu * torch.sign(flat[idx])
        index_bits = k * max(1.0, np.ceil(np.log2(max(n, 2))))
        return out.view_as(t), k * 1.0 + index_bits + FP32_BITS   # sign bit + idx + mu


class CountSketch(Compressor):
    """FetchSGD-style Count-Sketch compression of the update vector."""

    def __init__(self, width_ratio: float = 0.1, depth: int = 3, seed: int = 0):
        self.wr, self.depth, self.seed = width_ratio, depth, seed
        self.name = f"sketch{int(width_ratio*100)}"

    def __call__(self, t):
        n = t.numel()
        if n == 0:
            return t, 0.0
        w = max(1, int(n * self.wr))
        g = torch.Generator(device="cpu").manual_seed(self.seed + n)
        flat = t.flatten().cpu()
        table = torch.zeros(self.depth, w)
        hashes, signs = [], []
        for d in range(self.depth):
            h = torch.randint(0, w, (n,), generator=g)
            s = (torch.randint(0, 2, (n,), generator=g).float() * 2 - 1)
            table[d].index_add_(0, h, flat * s)
            hashes.append(h); signs.append(s)
        est = torch.stack([signs[d] * table[d][hashes[d]] for d in range(self.depth)])
        rec = est.median(dim=0).values.view_as(t.cpu()).to(t.device)
        return rec, self.depth * w * FP32_BITS


def build_compressor(name: str | None, **kw) -> Compressor:
    name = (name or "none").lower()
    if name in ("none", "off"):
        return NoCompression()
    if name.startswith("quant"):
        return UniformQuantizer(int(kw.get("bits", name.replace("quant", "") or 8)))
    if name.startswith("topk"):
        return TopKSparsifier(float(kw.get("k_ratio", 0.1)))
    if name.startswith("stc"):
        return SparseTernary(float(kw.get("k_ratio", 0.05)))
    if name.startswith("sketch") or name == "countsketch":
        return CountSketch(float(kw.get("width_ratio", 0.1)), int(kw.get("depth", 3)))
    raise ValueError(f"unknown compressor '{name}'")

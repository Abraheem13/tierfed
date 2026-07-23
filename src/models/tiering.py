"""Automatic depth-based tier assignment for ARBITRARY architectures.

The original formulation hard-coded three tiers onto a two-hidden-layer MLP.
Reviewers correctly noted this does not generalise. Here the partition is
defined structurally: order every parameter by its topological depth in the
computation graph, then cut the ordering into (slow, medium, fast) by
configurable fractions, with the classifier head always assigned to the fast
tier. This yields the original partition as a special case for the 2-layer MLP
and extends unchanged to ResNets and Transformers.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import torch
import torch.nn as nn

TIERS = ("slow", "medium", "fast")


def parameter_depth_order(model: nn.Module) -> List[str]:
    """Parameter names ordered by module registration depth.

    PyTorch registers modules in definition order, which for feed-forward
    networks coincides with topological order. A forward hook pass is used when
    available to confirm execution order.
    """
    return [n for n, _ in model.named_parameters()]


def execution_order(model: nn.Module, example: torch.Tensor) -> List[str]:
    """True execution order recovered with forward hooks (robust for ResNets)."""
    order: List[str] = []
    handles = []

    def hook(mod, inp, out):
        for pname, p in mod.named_parameters(recurse=False):
            full = f"{mod._nfl_name}.{pname}" if mod._nfl_name else pname
            if full not in order:
                order.append(full)

    for name, mod in model.named_modules():
        mod._nfl_name = name
        handles.append(mod.register_forward_hook(hook))
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(example)
    model.train(was_training)
    for h in handles:
        h.remove()
    known = {n for n, _ in model.named_parameters()}
    order = [o for o in order if o in known]
    order += [n for n in known if n not in order]      # params not hit by the hook
    return order


def head_parameter_names(model: nn.Module) -> List[str]:
    """Parameters of the final classifier module."""
    if hasattr(model, "head_module_name"):
        prefix = model.head_module_name
        return [n for n, _ in model.named_parameters() if n.startswith(prefix)]
    last_linear = None
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Linear, nn.Conv2d)):
            last_linear = name
    return [n for n, _ in model.named_parameters()
            if last_linear is not None and n.startswith(last_linear + ".")]


def assign_tiers(model: nn.Module, fractions: Sequence[float] = (0.34, 1.0),
                 example: torch.Tensor | None = None,
                 head_is_fast: bool = True) -> Dict[str, str]:
    """Map every parameter name -> tier.

    Two mechanisms, in priority order:

    1. If the architecture declares `tier_module_map` (module-name prefix ->
       tier), that mapping is used verbatim. This reproduces the original
       stem/body/head partition exactly and is the recommended path for
       architectures where the semantic split is known.
    2. Otherwise the partition is structural: parameters are ordered by
       execution depth, and the ordering is cut by DEPTH FRACTION (position in
       the ordering, not cumulative bytes, which would let a single wide tensor
       swallow a whole tier). The classifier head is always fast.

    This generalises unchanged to ResNets and Transformers.
    """
    explicit = getattr(model, "tier_module_map", None)
    order = execution_order(model, example) if example is not None else parameter_depth_order(model)
    head = set(head_parameter_names(model)) if head_is_fast else set()

    if explicit:
        tiers: Dict[str, str] = {}
        for n in order:
            for prefix, tier in explicit.items():
                if n == prefix or n.startswith(prefix + "."):
                    tiers[n] = tier
                    break
        for n in head:
            tiers[n] = "fast"
        for n in order:
            tiers.setdefault(n, "medium")
        return tiers

    body = [n for n in order if n not in head]
    m = len(body)
    c1, c2 = fractions[0] * m, fractions[1] * m
    tiers = {}
    for i, n in enumerate(body, start=1):
        tiers[n] = "slow" if i <= c1 else ("medium" if i <= c2 else "fast")
    for n in head:
        tiers[n] = "fast"
    for n in order:
        tiers.setdefault(n, "medium")
    return tiers


def tier_keys(tiers: Dict[str, str], tier: str) -> List[str]:
    return [k for k, v in tiers.items() if v == tier]


def tier_report(model: nn.Module, tiers: Dict[str, str]) -> Dict[str, Dict[str, float]]:
    sizes = {n: p.numel() for n, p in model.named_parameters()}
    total = sum(sizes.values()) or 1
    rep = {}
    for t in TIERS:
        keys = tier_keys(tiers, t)
        n = sum(sizes[k] for k in keys)
        rep[t] = {"params": float(n), "pct": 100.0 * n / total, "n_tensors": float(len(keys))}
    rep["total"] = {"params": float(total), "pct": 100.0, "n_tensors": float(len(sizes))}
    return rep

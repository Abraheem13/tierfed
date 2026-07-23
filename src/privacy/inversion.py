"""Gradient-inversion attack used as an empirical privacy evaluation.

A DLG/iDLG-style optimisation attack reconstructs client inputs from the
transmitted update. It is run against each strategy so the paper can report a
measured attack success (reconstruction error / feature correlation) rather
than asserting privacy. The multi-frequency schedule is expected to help
because the head -- the layer carrying the most label-correlated signal -- is
never transmitted, and only a slice is exposed per round.
"""
from __future__ import annotations

from typing import Dict, List, Mapping

import numpy as np
import torch
import torch.nn as nn


def _flat(d: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([v.flatten() for v in d.values()])


def gradient_inversion_attack(model: nn.Module, target_state_delta: Mapping[str, torch.Tensor],
                              x_shape, n_samples: int, device, lr: float = 0.1,
                              steps: int = 300, seed: int = 0,
                              loss_fn: nn.Module | None = None) -> Dict[str, float]:
    """Reconstruct a batch from an observed parameter update.

    Only the parameters actually present in `target_state_delta` are matched --
    that is the point: a strategy that transmits fewer tensors gives the
    attacker a weaker signal.
    """
    torch.manual_seed(seed)
    loss_fn = loss_fn or nn.CrossEntropyLoss()
    keys = list(target_state_delta.keys())
    target = _flat({k: target_state_delta[k].to(device) for k in keys}).detach()
    x = torch.randn(n_samples, *x_shape, device=device, requires_grad=True)
    y = torch.randint(0, 2, (n_samples,), device=device)
    opt = torch.optim.Adam([x], lr=lr)
    params = {n: p for n, p in model.named_parameters()}
    sel = [params[k] for k in keys if k in params]
    best = float("inf")
    for _ in range(steps):
        opt.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        grads = torch.autograd.grad(loss, sel, create_graph=True, allow_unused=True)
        g = torch.cat([(gr if gr is not None else torch.zeros_like(p)).flatten()
                       for gr, p in zip(grads, sel)])
        n = min(g.numel(), target.numel())
        obj = ((g[:n] - target[:n]) ** 2).sum()
        obj.backward()
        opt.step()
        best = min(best, float(obj.detach()))
    return {"x_recon": x.detach(), "match_objective": best}


def evaluate_inversion(model: nn.Module, update: Mapping[str, torch.Tensor],
                       x_true: torch.Tensor, device, steps: int = 300,
                       seed: int = 0) -> Dict[str, float]:
    """Quantifies how much of the true batch the attacker recovers."""
    res = gradient_inversion_attack(model, update, tuple(x_true.shape[1:]),
                                    x_true.shape[0], device, steps=steps, seed=seed)
    xr = res["x_recon"].flatten(1).cpu().numpy()
    xt = x_true.flatten(1).cpu().numpy()
    # Best-matching permutation is unknown; use the closest reconstruction per true row.
    d = ((xt[:, None, :] - xr[None, :, :]) ** 2).mean(-1)
    j = d.argmin(1)
    mse = float(d[np.arange(len(xt)), j].mean())
    denom = float((xt ** 2).mean()) or 1.0
    corr = []
    for i, jj in enumerate(j):
        a, b = xt[i], xr[jj]
        if a.std() > 1e-8 and b.std() > 1e-8:
            corr.append(float(np.corrcoef(a, b)[0, 1]))
    return {
        "inversion_mse": mse,
        "inversion_nmse": mse / denom,
        "inversion_feature_corr": float(np.mean(corr)) if corr else float("nan"),
        "match_objective": res["match_objective"],
        "n_tensors_exposed": float(len(update)),
        "n_params_exposed": float(sum(v.numel() for v in update.values())),
    }

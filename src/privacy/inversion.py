"""Update-inversion attack as an empirical attack-surface measurement.

v2 ran a DLG-style attack on a single-batch gradient at initialisation and on
the fixed schedule, then labelled it TierFed. v3 attacks the slice a client
ACTUALLY transmits (its post-local-training update) at synchronisation rounds,
for every strategy, inside the training loop (see FederatedServer._attack).

The attacker knows the broadcast model and the transmitted update delta over
the transmitted keys only. It optimises a candidate batch x so that the
gradient of the loss on x, restricted to those keys, matches the direction of
the update (cosine objective, Geiping et al. 2020, with total-variation prior
omitted for tabular inputs). Because the update is a multi-step Adam update
rather than a single gradient, this is the standard "update inversion" setting
of Geiping et al.; success is reported as normalised MSE and the feature
correlation of the best-matching reconstruction, so lower correlation means a
smaller attack surface. The attack is intentionally generic: the point is a
like-for-like comparison of what each strategy exposes, not the strongest
possible adversary.
"""
from __future__ import annotations

from typing import Dict, Mapping

import numpy as np
import torch
import torch.nn as nn


def _flat(d: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([v.flatten() for v in d.values()])


def attack_update(model: nn.Module, delta: Mapping[str, torch.Tensor], x_true: torch.Tensor,
                  device, lr: float = 0.05, steps: int = 300, seed: int = 0,
                  label_known: bool = False) -> Dict[str, float]:
    keys = [k for k in delta if k in dict(model.named_parameters())]
    if not keys:
        return {"inversion_nmse": float("nan"), "inversion_feature_corr": float("nan"),
                "match_objective": float("nan")}
    torch.manual_seed(seed)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(True)
    params = dict(model.named_parameters())
    sel = [params[k] for k in keys]
    # the update direction is minus the gradient direction (descent)
    target = -_flat({k: delta[k].to(device) for k in keys}).detach()
    n, shape = x_true.shape[0], tuple(x_true.shape[1:])
    x = torch.randn(n, *shape, device=device, requires_grad=True)
    y = torch.randint(0, 2, (n,), device=device)
    opt = torch.optim.Adam([x], lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    best = float("inf")
    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        grads = torch.autograd.grad(loss, sel, create_graph=True, allow_unused=True)
        g = torch.cat([(gr if gr is not None else torch.zeros_like(p)).flatten()
                       for gr, p in zip(grads, sel)])
        obj = 1.0 - torch.nn.functional.cosine_similarity(g, target, dim=0)
        obj.backward()
        opt.step()
        best = min(best, float(obj.detach()))
    xr = x.detach().flatten(1).cpu().numpy()
    xt = x_true.flatten(1).cpu().numpy()
    d = ((xt[:, None, :] - xr[None, :, :]) ** 2).mean(-1)
    j = d.argmin(1)
    mse = float(d[np.arange(len(xt)), j].mean())
    denom = float((xt ** 2).mean()) or 1.0
    corr = []
    for i, jj in enumerate(j):
        a, b = xt[i], xr[jj]
        if a.std() > 1e-8 and b.std() > 1e-8:
            corr.append(float(np.corrcoef(a, b)[0, 1]))
    return {"inversion_nmse": mse / denom,
            "inversion_feature_corr": float(np.mean(corr)) if corr else float("nan"),
            "match_objective": best}

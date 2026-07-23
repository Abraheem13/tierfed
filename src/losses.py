"""Local objectives, including advanced class-imbalance losses requested by
reviewers beyond plain inverse-frequency reweighting."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedCE(nn.Module):
    """Inverse-frequency class-reweighted cross entropy (original baseline)."""
    def __init__(self, class_weights: torch.Tensor | None = None):
        super().__init__()
        self.w = class_weights
    def forward(self, logits, target):
        w = self.w.to(logits.device) if self.w is not None else None
        return F.cross_entropy(logits, target, weight=w)


class FocalLoss(nn.Module):
    """Lin et al. focal loss; down-weights easy negatives under heavy skew."""
    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None):
        super().__init__()
        self.gamma, self.alpha = gamma, alpha
    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=1)
        logpt = logp.gather(1, target.view(-1, 1)).squeeze(1)
        pt = logpt.exp()
        loss = -((1 - pt) ** self.gamma) * logpt
        if self.alpha is not None:
            loss = loss * self.alpha.to(logits.device).gather(0, target)
        return loss.mean()


class LDAMLoss(nn.Module):
    """Label-Distribution-Aware Margin loss (Cao et al.) for rare positives."""
    def __init__(self, cls_num_list, max_m: float = 0.5, s: float = 30.0, weight=None):
        super().__init__()
        m = 1.0 / np.sqrt(np.sqrt(np.maximum(np.asarray(cls_num_list, float), 1.0)))
        m = m * (max_m / m.max())
        self.m_list = torch.tensor(m, dtype=torch.float32)
        self.s, self.weight = s, weight
    def forward(self, logits, target):
        idx = torch.zeros_like(logits, dtype=torch.bool)
        idx.scatter_(1, target.view(-1, 1), True)
        margins = self.m_list.to(logits.device)[target].view(-1, 1)
        logits_m = torch.where(idx, logits - margins, logits)
        w = self.weight.to(logits.device) if self.weight is not None else None
        return F.cross_entropy(self.s * logits_m, target, weight=w)


class ClassBalancedCE(nn.Module):
    """Cui et al. class-balanced loss using effective-number reweighting."""
    def __init__(self, cls_num_list, beta: float = 0.999):
        super().__init__()
        counts = np.maximum(np.asarray(cls_num_list, float), 1.0)
        eff = (1.0 - np.power(beta, counts)) / (1.0 - beta)
        w = 1.0 / eff
        w = w / w.sum() * len(counts)
        self.w = torch.tensor(w, dtype=torch.float32)
    def forward(self, logits, target):
        return F.cross_entropy(logits, target, weight=self.w.to(logits.device))


def build_loss(name: str, cls_num_list, device, focal_gamma: float = 2.0, cb_beta: float = 0.999):
    """Factory; `cls_num_list` holds the per-class counts on the local client."""
    counts = np.maximum(np.asarray(cls_num_list, float), 1.0)
    inv = torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float32)
    name = (name or "weighted_ce").lower()
    if name in ("ce", "cross_entropy"):
        return WeightedCE(None).to(device)
    if name in ("weighted_ce", "inverse_frequency"):
        return WeightedCE(inv).to(device)
    if name == "focal":
        return FocalLoss(gamma=focal_gamma, alpha=inv).to(device)
    if name == "ldam":
        return LDAMLoss(counts, weight=inv).to(device)
    if name in ("cb", "class_balanced"):
        return ClassBalancedCE(counts, beta=cb_beta).to(device)
    raise ValueError(f"unknown loss '{name}'")

"""Architectures: configurable-depth MLP, 1-D tabular ResNet, FT-Transformer,
and a small CNN/ResNet for imaging. All expose `head_module_name` so the
tiering rules apply uniformly."""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class NestedMLP(nn.Module):
    """Configurable-depth MLP (the original 93-128-128-64-2 is the default)."""
    head_module_name = "head"
    tier_module_map = {"stem": "slow", "body": "medium", "head": "fast"}

    def __init__(self, in_dim: int, hidden: Sequence[int] = (128, 128, 64),
                 n_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.stem = nn.Sequential(nn.Linear(in_dim, hidden[0]), nn.ReLU(), nn.Dropout(dropout))
        blocks = []
        for a, b in zip(hidden[:-1], hidden[1:]):
            blocks += [nn.Linear(a, b), nn.ReLU(), nn.Dropout(dropout)]
        self.body = nn.Sequential(*blocks)
        self.head = nn.Linear(hidden[-1], n_classes)

    def forward(self, x):
        return self.head(self.body(self.stem(x.flatten(1))))


class _ResBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1, self.fc2 = nn.Linear(dim, dim * 2), nn.Linear(dim * 2, dim)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        return x + self.drop(self.fc2(torch.relu(self.fc1(self.norm(x)))))


class TabularResNet(nn.Module):
    """Deeper tabular backbone; verifies tiering beyond a shallow MLP."""
    head_module_name = "head"
    tier_module_map = {"stem": "slow", "body": "medium", "norm": "medium", "head": "fast"}

    def __init__(self, in_dim: int, dim: int = 128, depth: int = 6,
                 n_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        self.stem = nn.Linear(in_dim, dim)
        self.body = nn.Sequential(*[_ResBlock(dim, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, n_classes)

    def forward(self, x):
        return self.head(self.norm(self.body(self.stem(x.flatten(1)))))


class FTTransformer(nn.Module):
    """Feature-tokeniser Transformer for tabular data (Gorishniy et al.)."""
    head_module_name = "head"
    tier_module_map = {"tokeniser": "slow", "bias": "slow", "cls": "slow",
                       "body": "medium", "norm": "medium", "head": "fast"}

    def __init__(self, in_dim: int, dim: int = 64, depth: int = 3, heads: int = 8,
                 n_classes: int = 2, dropout: float = 0.1):
        super().__init__()
        self.tokeniser = nn.Parameter(torch.randn(in_dim, dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(in_dim, dim))
        self.cls = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 2, dropout,
                                           batch_first=True, norm_first=True)
        self.body = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, n_classes)

    def forward(self, x):
        x = x.flatten(1)
        tok = x.unsqueeze(-1) * self.tokeniser.unsqueeze(0) + self.bias.unsqueeze(0)
        tok = torch.cat([self.cls.expand(x.size(0), -1, -1), tok], dim=1)
        return self.head(self.norm(self.body(tok)[:, 0]))


class SmallCNN(nn.Module):
    """Compact CNN for MedMNIST-scale imaging."""
    head_module_name = "head"
    tier_module_map = {"stem": "slow", "body": "medium", "head": "fast"}

    def __init__(self, in_ch: int = 1, width: int = 32, n_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o),
                                 nn.ReLU(), nn.Conv2d(o, o, 3, padding=1),
                                 nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
        self.stem = blk(in_ch, width)
        self.body = nn.Sequential(blk(width, width * 2), blk(width * 2, width * 4))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(width * 4, n_classes)

    def forward(self, x):
        h = self.pool(self.body(self.stem(x))).flatten(1)
        return self.head(self.drop(h))


def build_model(name: str, input_shape, n_classes: int = 2, **kw) -> nn.Module:
    in_dim = int(torch.tensor(list(input_shape)).prod().item())
    name = (name or "mlp").lower()
    if name == "mlp":
        return NestedMLP(in_dim, kw.get("hidden", (128, 128, 64)), n_classes, kw.get("dropout", 0.3))
    if name in ("resnet", "tabresnet"):
        return TabularResNet(in_dim, kw.get("dim", 128), kw.get("depth", 6), n_classes, kw.get("dropout", 0.2))
    if name in ("transformer", "ft_transformer"):
        return FTTransformer(in_dim, kw.get("dim", 64), kw.get("depth", 3), kw.get("heads", 8),
                             n_classes, kw.get("dropout", 0.1))
    if name in ("cnn", "smallcnn"):
        return SmallCNN(int(input_shape[0]), kw.get("width", 32), n_classes, kw.get("dropout", 0.2))
    raise ValueError(f"unknown model '{name}'")

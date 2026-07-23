"""Dataset abstraction shared by every corpus + a registry.

A FederatedDataset is: a global train/test tensor pair, a list of client index
arrays, and metadata. Adding a new corpus means adding one loader function
and registering it -- this is what lets the same experiment code run over
tabular EHR, imaging, and synthetic covariate-shift benchmarks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class FederatedDataset:
    name: str
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    client_indices: List[np.ndarray]
    meta: Dict = field(default_factory=dict)

    @property
    def n_features(self) -> int:
        return int(np.prod(self.x_train.shape[1:]))

    @property
    def n_clients(self) -> int:
        return len(self.client_indices)

    @property
    def input_shape(self):
        return tuple(self.x_train.shape[1:])

    def client_counts(self, k: int) -> List[int]:
        y = self.y_train[self.client_indices[k]]
        return [int((y == c).sum()) for c in (0, 1)]

    def client_loader(self, k: int, batch_size: int = 64, shuffle: bool = True,
                      balanced: bool = False, seed: int = 0) -> DataLoader:
        idx = self.client_indices[k]
        x = torch.as_tensor(self.x_train[idx], dtype=torch.float32)
        y = torch.as_tensor(self.y_train[idx], dtype=torch.long)
        ds = TensorDataset(x, y)
        g = torch.Generator().manual_seed(seed + k)
        if balanced:
            # Federated re-balancing sampler: an oversampling alternative to
            # loss reweighting, comparable to a local federated-SMOTE variant.
            from torch.utils.data import WeightedRandomSampler
            counts = np.bincount(y.numpy(), minlength=2).astype(float)
            w = 1.0 / np.maximum(counts, 1.0)
            sw = torch.as_tensor(w[y.numpy()], dtype=torch.double)
            sampler = WeightedRandomSampler(sw, num_samples=len(sw), replacement=True, generator=g)
            return DataLoader(ds, batch_size=batch_size, sampler=sampler, drop_last=False)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g, drop_last=False)

    def test_loader(self, batch_size: int = 512) -> DataLoader:
        x = torch.as_tensor(self.x_test, dtype=torch.float32)
        y = torch.as_tensor(self.y_test, dtype=torch.long)
        return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)

    def client_test_split(self, frac: float = 0.2, seed: int = 0):
        """Per-client held-out split, needed to evaluate PERSONALISED models
        (a global test set cannot measure a per-client head)."""
        rng = np.random.default_rng(seed)
        tr, te = [], []
        for idx in self.client_indices:
            perm = rng.permutation(len(idx))
            cut = max(1, int(len(idx) * frac))
            te.append(idx[perm[:cut]]); tr.append(idx[perm[cut:]])
        return tr, te

    def describe(self) -> Dict:
        return {
            "dataset": self.name,
            "n_train": int(len(self.y_train)), "n_test": int(len(self.y_test)),
            "n_features": self.n_features, "input_shape": list(self.input_shape),
            "n_clients": self.n_clients,
            "prevalence_train": float(self.y_train.mean()),
            "prevalence_test": float(self.y_test.mean()),
            **{k: v for k, v in self.meta.items() if not isinstance(v, (np.ndarray, list))},
        }


REGISTRY: Dict[str, Callable[..., FederatedDataset]] = {}


def register(name: str):
    def deco(fn):
        REGISTRY[name] = fn
        return fn
    return deco


def load_dataset(name: str, **kwargs) -> FederatedDataset:
    if name not in REGISTRY:
        raise KeyError(f"unknown dataset '{name}'. available: {sorted(REGISTRY)}")
    return REGISTRY[name](**kwargs)


def available_datasets() -> List[str]:
    return sorted(REGISTRY)

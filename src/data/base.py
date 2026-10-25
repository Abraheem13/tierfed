"""Dataset abstraction shared by every corpus + a registry.

A FederatedDataset holds a global train/test tensor pair, a list of client
index arrays, an optional `groups` array (patient identifier) so that every
split is patient-grouped, and metadata.
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
    groups: Optional[np.ndarray] = None       # patient id per train row (None = each row its own)

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

    def loader_for(self, idx: np.ndarray, batch_size: int = 64, shuffle: bool = True,
                   balanced: bool = False, seed: int = 0) -> DataLoader:
        x = torch.as_tensor(self.x_train[idx], dtype=torch.float32)
        y = torch.as_tensor(self.y_train[idx], dtype=torch.long)
        ds = TensorDataset(x, y)
        g = torch.Generator().manual_seed(seed)
        if balanced:
            from torch.utils.data import WeightedRandomSampler
            counts = np.bincount(y.numpy(), minlength=2).astype(float)
            w = 1.0 / np.maximum(counts, 1.0)
            sw = torch.as_tensor(w[y.numpy()], dtype=torch.double)
            sampler = WeightedRandomSampler(sw, num_samples=len(sw), replacement=True, generator=g)
            return DataLoader(ds, batch_size=batch_size, sampler=sampler, drop_last=False)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g, drop_last=False)

    def client_loader(self, k: int, **kw) -> DataLoader:
        return self.loader_for(self.client_indices[k], **kw)

    def client_test_split(self, frac: float = 0.2, seed: int = 0):
        """Per-client held-out split, PATIENT-GROUPED when `groups` is given,
        needed to evaluate personalised models without patient leakage."""
        rng = np.random.default_rng(seed)
        tr, te = [], []
        for idx in self.client_indices:
            if self.groups is None:
                perm = rng.permutation(len(idx))
                cut = max(1, int(len(idx) * frac))
                te.append(idx[perm[:cut]]); tr.append(idx[perm[cut:]])
                continue
            g = self.groups[idx]
            uniq = np.unique(g)
            rng.shuffle(uniq)
            n_te = max(1, int(len(uniq) * frac))
            te_groups = set(uniq[:n_te].tolist())
            mask = np.array([gi in te_groups for gi in g])
            te.append(idx[mask]); tr.append(idx[~mask])
        return tr, te

    def describe(self) -> Dict:
        return {
            "dataset": self.name,
            "n_train": int(len(self.y_train)), "n_test": int(len(self.y_test)),
            "n_features": self.n_features, "input_shape": list(self.input_shape),
            "n_clients": self.n_clients,
            "n_patients_train": int(len(np.unique(self.groups))) if self.groups is not None else int(len(self.y_train)),
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

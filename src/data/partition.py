"""Client partitioning schemes.

Reviewers objected that partitioning by medical specialty is a synthetic
proxy for institutional heterogeneity. This module therefore provides
(a) *natural* partitioning by a true site identifier column (eICU/MIMIC),
(b) Dirichlet label-skew partitioning with a controllable alpha, the standard
    non-IID protocol in the FL literature, and
(c) the original specialty/LPT scheme, retained for backwards comparability.
Every scheme returns index lists and a diagnostic heterogeneity report.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def _report(y: np.ndarray, parts: List[np.ndarray]) -> Dict[str, float]:
    """Quantifies how non-IID a partition actually is."""
    sizes = np.array([len(p) for p in parts], dtype=float)
    prevs = np.array([float(y[p].mean()) if len(p) else np.nan for p in parts])
    # Global label distribution vs per-client, as a mean total-variation distance.
    g = np.array([1 - y.mean(), y.mean()])
    tvs = []
    for p in parts:
        if len(p) == 0:
            continue
        loc = np.array([1 - y[p].mean(), y[p].mean()])
        tvs.append(0.5 * np.abs(loc - g).sum())
    return {
        "n_clients": float(len(parts)),
        "size_min": float(sizes.min()), "size_median": float(np.median(sizes)),
        "size_max": float(sizes.max()),
        "size_gini": float(_gini(sizes)),
        "prevalence_min": float(np.nanmin(prevs)), "prevalence_max": float(np.nanmax(prevs)),
        "prevalence_std": float(np.nanstd(prevs)),
        "label_tv_mean": float(np.mean(tvs)) if tvs else float("nan"),
    }


def _gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def natural_sites(site_ids: Sequence, min_size: int = 50, max_clients: int | None = None):
    """Partition by a REAL site identifier (e.g. eICU hospitalid).

    Sites smaller than `min_size` are merged into a pooled 'small-sites' client
    rather than discarded, so no data is lost.
    """
    site_ids = np.asarray(site_ids)
    parts, small = [], []
    order = sorted({s for s in site_ids.tolist()}, key=lambda s: -int((site_ids == s).sum()))
    for s in order:
        idx = np.where(site_ids == s)[0]
        (parts if len(idx) >= min_size else small).append(idx)
    if small:
        parts.append(np.concatenate(small))
    if max_clients is not None and len(parts) > max_clients:
        keep, tail = parts[: max_clients - 1], parts[max_clients - 1:]
        parts = keep + [np.concatenate(tail)]
    return parts


def dirichlet_label_skew(y: Sequence[int], n_clients: int, alpha: float, seed: int = 0,
                         min_size: int = 20) -> List[np.ndarray]:
    """Standard Dirichlet non-IID protocol: lower alpha => more label skew."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    n_classes = int(y.max()) + 1
    for _ in range(100):
        parts = [[] for _ in range(n_clients)]
        for c in range(n_classes):
            idx = np.where(y == c)[0]
            rng.shuffle(idx)
            p = rng.dirichlet(np.repeat(alpha, n_clients))
            cuts = (np.cumsum(p) * len(idx)).astype(int)[:-1]
            for i, chunk in enumerate(np.split(idx, cuts)):
                parts[i].extend(chunk.tolist())
        parts = [np.array(sorted(p), dtype=int) for p in parts]
        if min(len(p) for p in parts) >= min_size:
            return parts
    return parts


def quantity_skew(n: int, n_clients: int, alpha: float = 2.0, seed: int = 0) -> List[np.ndarray]:
    """Unequal client sizes (size heterogeneity without label skew)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    p = rng.dirichlet(np.repeat(alpha, n_clients))
    cuts = (np.cumsum(p) * n).astype(int)[:-1]
    return [np.array(sorted(c), dtype=int) for c in np.split(idx, cuts)]


def lpt_group_partition(groups: Sequence, n_clients: int) -> List[np.ndarray]:
    """Original scheme: longest-processing-time bucketing of a group column
    (e.g. medical specialty), keeping each group intact within one client."""
    groups = np.asarray(groups)
    uniq = sorted({g for g in groups.tolist()}, key=lambda g: -int((groups == g).sum()))
    buckets: List[List[int]] = [[] for _ in range(n_clients)]
    loads = np.zeros(n_clients)
    for g in uniq:
        idx = np.where(groups == g)[0]
        j = int(np.argmin(loads))
        buckets[j].extend(idx.tolist())
        loads[j] += len(idx)
    return [np.array(sorted(b), dtype=int) for b in buckets]


def build_partition(scheme: str, y, n_clients: int, seed: int = 0, site_ids=None,
                    groups=None, alpha: float = 0.5, min_size: int = 50,
                    verbose: bool = True):
    """Dispatch + heterogeneity diagnostics."""
    y = np.asarray(y).astype(int)
    scheme = scheme.lower()
    if scheme in ("natural", "site", "hospital"):
        if site_ids is None:
            raise ValueError("natural partition needs site_ids")
        parts = natural_sites(site_ids, min_size=min_size, max_clients=n_clients)
    elif scheme in ("dirichlet", "label_skew"):
        parts = dirichlet_label_skew(y, n_clients, alpha, seed)
    elif scheme in ("quantity", "size_skew"):
        parts = quantity_skew(len(y), n_clients, alpha, seed)
    elif scheme in ("lpt", "specialty"):
        if groups is None:
            raise ValueError("lpt partition needs groups")
        parts = lpt_group_partition(groups, n_clients)
    elif scheme == "iid":
        rng = np.random.default_rng(seed)
        parts = [np.array(sorted(c), int) for c in np.array_split(rng.permutation(len(y)), n_clients)]
    else:
        raise ValueError(f"unknown partition scheme '{scheme}'")
    parts = [p for p in parts if len(p) > 0]
    rep = _report(y, parts)
    rep["scheme"] = scheme
    if verbose:
        print(f"[partition] {scheme}: {len(parts)} clients | sizes "
              f"{rep['size_min']:.0f}/{rep['size_median']:.0f}/{rep['size_max']:.0f} "
              f"| prevalence {rep['prevalence_min']:.3f}-{rep['prevalence_max']:.3f} "
              f"| label TV {rep['label_tv_mean']:.4f}")
    return parts, rep

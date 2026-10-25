"""Client partitioning schemes with heterogeneity diagnostics.

(a) natural: a true site identifier (eICU hospitalid, MIMIC admission location)
(b) dirichlet: label-skew with alpha, with a minimum client size enforced for
    every client (v2 produced 1-sample clients at alpha = 0.1)
(c) lpt: longest-processing-time bucketing of a group column (medical
    specialty) with a CLIENT-SHARE CAP: a group larger than `max_share` of the
    data is split (patient-grouped) before bucketing, so no single client can
    dominate the federation (in v2 the "missing specialty" bucket held 49%).
(d) quantity, iid.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def _gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=float)); n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def _report(y: np.ndarray, parts: List[np.ndarray]) -> Dict[str, float]:
    sizes = np.array([len(p) for p in parts], dtype=float)
    prevs = np.array([float(y[p].mean()) if len(p) else np.nan for p in parts])
    g = np.array([1 - y.mean(), y.mean()])
    tvs = [0.5 * np.abs(np.array([1 - y[p].mean(), y[p].mean()]) - g).sum() for p in parts if len(p)]
    single = int(sum(1 for p in parts if len(p) and len(np.unique(y[p])) < 2))
    return {"n_clients": float(len(parts)),
            "size_min": float(sizes.min()), "size_median": float(np.median(sizes)),
            "size_max": float(sizes.max()), "size_max_share": float(sizes.max() / sizes.sum()),
            "size_gini": _gini(sizes),
            "prevalence_min": float(np.nanmin(prevs)), "prevalence_max": float(np.nanmax(prevs)),
            "prevalence_std": float(np.nanstd(prevs)),
            "label_tv_mean": float(np.mean(tvs)) if tvs else float("nan"),
            "n_single_class_clients": float(single)}


def natural_sites(site_ids: Sequence, min_size: int = 50, max_clients: int | None = None):
    site_ids = np.asarray(site_ids)
    parts, small = [], []
    order = sorted(set(site_ids.tolist()), key=lambda s: -int((site_ids == s).sum()))
    for s in order:
        idx = np.where(site_ids == s)[0]
        (parts if len(idx) >= min_size else small).append(idx)
    if small:
        parts.append(np.concatenate(small))
    if max_clients is not None and len(parts) > max_clients:
        parts = parts[: max_clients - 1] + [np.concatenate(parts[max_clients - 1:])]
    return parts


def dirichlet_label_skew(y, n_clients: int, alpha: float, seed: int = 0,
                         min_size: int = 20, groups=None, max_tries: int = 500) -> List[np.ndarray]:
    """Standard Dirichlet protocol; retries until every client has >= min_size
    rows. If `groups` is given, whole patients are assigned together."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    units = np.arange(len(y)) if groups is None else np.unique(groups)
    unit_label = y if groups is None else np.array([int(round(y[groups == u].mean())) for u in units])
    n_classes = int(y.max()) + 1
    best, best_min = None, -1
    for _ in range(max_tries):
        parts = [[] for _ in range(n_clients)]
        for c in range(n_classes):
            idx = np.where(unit_label == c)[0]
            rng.shuffle(idx)
            p = rng.dirichlet(np.repeat(alpha, n_clients))
            cuts = (np.cumsum(p) * len(idx)).astype(int)[:-1]
            for i, chunk in enumerate(np.split(idx, cuts)):
                parts[i].extend(chunk.tolist())
        if groups is not None:
            parts = [np.where(np.isin(groups, units[p]))[0] for p in parts]
        parts = [np.array(sorted(p), dtype=int) for p in parts]
        m = min(len(p) for p in parts)
        if m >= min_size:
            return parts
        if m > best_min:
            best, best_min = parts, m
    return [p for p in best if len(p) >= min_size]


def quantity_skew(n: int, n_clients: int, alpha: float = 2.0, seed: int = 0) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    p = rng.dirichlet(np.repeat(alpha, n_clients))
    cuts = (np.cumsum(p) * n).astype(int)[:-1]
    return [np.array(sorted(c), dtype=int) for c in np.split(idx, cuts)]


def lpt_group_partition(group_col: Sequence, n_clients: int, max_share: float = 0.15,
                        patient_groups=None, seed: int = 0) -> List[np.ndarray]:
    """LPT bucketing of a categorical column with a client-share cap.

    Categories whose size exceeds `max_share` of all rows are split into
    chunks of at most that size (patient-grouped when `patient_groups` is
    given) before bucketing, so the largest client holds at most roughly
    max_share + the largest remaining category.
    """
    rng = np.random.default_rng(seed)
    gcol = np.asarray(group_col)
    n = len(gcol)
    cap = max(1, int(max_share * n))
    pieces: List[np.ndarray] = []
    for g in sorted(set(gcol.tolist()), key=lambda s: -int((gcol == s).sum())):
        idx = np.where(gcol == g)[0]
        if len(idx) <= cap:
            pieces.append(idx); continue
        if patient_groups is None:
            rng.shuffle(idx)
            pieces += [c for c in np.array_split(idx, int(np.ceil(len(idx) / cap)))]
        else:
            pg = np.asarray(patient_groups)[idx]
            uniq = np.unique(pg); rng.shuffle(uniq)
            n_chunks = int(np.ceil(len(idx) / cap))
            for ch in np.array_split(uniq, n_chunks):
                pieces.append(idx[np.isin(pg, ch)])
    pieces.sort(key=lambda p: -len(p))
    buckets: List[List[int]] = [[] for _ in range(n_clients)]
    loads = np.zeros(n_clients)
    for p in pieces:
        j = int(np.argmin(loads))
        buckets[j].extend(p.tolist()); loads[j] += len(p)
    return [np.array(sorted(b), dtype=int) for b in buckets if b]


def build_partition(scheme: str, y, n_clients: int, seed: int = 0, site_ids=None,
                    groups=None, patient_groups=None, alpha: float = 0.5, min_size: int = 50,
                    max_share: float = 0.15, verbose: bool = True):
    y = np.asarray(y).astype(int)
    scheme = scheme.lower()
    if scheme in ("natural", "site", "hospital"):
        if site_ids is None:
            raise ValueError("natural partition needs site_ids")
        parts = natural_sites(site_ids, min_size=min_size, max_clients=n_clients)
    elif scheme in ("dirichlet", "label_skew"):
        parts = dirichlet_label_skew(y, n_clients, alpha, seed, min_size=min_size, groups=patient_groups)
    elif scheme in ("quantity", "size_skew"):
        parts = quantity_skew(len(y), n_clients, alpha, seed)
    elif scheme in ("lpt", "specialty"):
        if groups is None:
            raise ValueError("lpt partition needs groups")
        parts = lpt_group_partition(groups, n_clients, max_share=max_share,
                                    patient_groups=patient_groups, seed=seed)
    elif scheme == "iid":
        rng = np.random.default_rng(seed)
        parts = [np.array(sorted(c), int) for c in np.array_split(rng.permutation(len(y)), n_clients)]
    else:
        raise ValueError(f"unknown partition scheme '{scheme}'")
    parts = [p for p in parts if len(p) > 0]
    rep = _report(y, parts); rep["scheme"] = scheme
    if verbose:
        print(f"[partition] {scheme}: {len(parts)} clients | sizes "
              f"{rep['size_min']:.0f}/{rep['size_median']:.0f}/{rep['size_max']:.0f} "
              f"(max share {rep['size_max_share']:.2f}) | prevalence "
              f"{rep['prevalence_min']:.3f}-{rep['prevalence_max']:.3f} | label TV {rep['label_tv_mean']:.4f} "
              f"| single-class clients {rep['n_single_class_clients']:.0f}")
    return parts, rep

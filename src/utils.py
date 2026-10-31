"""Seeding, determinism, devices, config handling, and small shared helpers."""
from __future__ import annotations

import json, logging, os, random, time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed every RNG that can influence a run and force deterministic kernels.

    v2 runs with identical configuration differed by up to 0.006 AUROC, which
    is larger than the effect the paper tests. v3 therefore (i) seeds all
    generators, (ii) selects deterministic cuBLAS/cuDNN algorithms, and
    (iii) pins intra-op threads to one so reduction order is fixed. The
    `test_determinism` unit test asserts bit-identical results.
    """
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        # fused attention kernels (flash / memory-efficient) are non-deterministic
        # on CUDA; force the math kernel so transformer runs are reproducible
        try:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
        except AttributeError:
            pass
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:          # very old torch
            torch.use_deterministic_algorithms(True)
        torch.set_num_threads(1)


def get_device(prefer: str = "auto") -> torch.device:
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer in ("cuda", "auto") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_logger(name: str = "tierfed", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", "%H:%M:%S"))
        logger.addHandler(h)
    logger.setLevel(level); logger.propagate = False
    return logger


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter(); return self
    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self.t0


def to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return {k: to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_json(path, payload) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(to_jsonable(payload), f, indent=2)


def load_yaml(path) -> Dict[str, Any]:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def state_numel(state: Mapping[str, torch.Tensor]) -> int:
    return int(sum(v.numel() for v in state.values()))


def clone_state(state: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in state.items()}


def flat_norm(state: Mapping[str, torch.Tensor]) -> float:
    if not state:
        return 0.0
    return float(torch.sqrt(sum((v.detach().float() ** 2).sum() for v in state.values())))

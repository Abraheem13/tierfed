from .base import FederatedDataset, available_datasets, load_dataset, register
from . import loaders  # noqa: F401  (registers all corpora)
from .partition import build_partition

__all__ = ["FederatedDataset", "load_dataset", "available_datasets", "register", "build_partition"]

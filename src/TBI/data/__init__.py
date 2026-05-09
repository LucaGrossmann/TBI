"""TBI dataset registry — unified loaders for benchmark datasets."""

from .registry import (
    DatasetSpec,
    load_dataset,
    list_datasets,
    register_dataset,
    DATASETS,
)

__all__ = [
    "DatasetSpec",
    "load_dataset",
    "list_datasets",
    "register_dataset",
    "DATASETS",
]

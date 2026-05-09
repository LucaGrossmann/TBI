"""
Unified dataset registry for benchmarking.

Provides a single entry point to load any registered dataset:

    from TBI.data import load_dataset, list_datasets
    X, b, meta = load_dataset("suez2018")

All loaders return ``(X, b, metadata)`` where:
    X        : (m, p, n) tensor
    b        : 1D array of block start indices
    metadata : dict with at least 'name', 'is_longitudinal', 'is_multi_omic'
"""

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np


@dataclass
class DatasetSpec:
    """Metadata describing a registered dataset."""
    name: str
    loader: Callable
    description: str
    is_longitudinal: bool
    is_multi_omic: bool
    has_groups: bool = False
    shape_hint: str = ""
    citation: str = ""
    is_primary: bool = True


# ---------------------------------------------------------------------------
# Lazy loaders (avoid import-time side effects)
# ---------------------------------------------------------------------------

def _load_cmipb():
    from TBI.data.cmipb import build_cmipb_tensor
    X, b, meta = build_cmipb_tensor(
        days=[0, 1, 3, 14],
        assays=["ab_titer", "cell_freq", "cytokine_olink", "cytokine_legendplex"],
    )
    meta.update({
        "name": "cmipb",
        "is_longitudinal": True,
        "is_multi_omic": True,
        "has_groups": True,
    })
    return X, b, meta


def _load_suez2018():
    from TBI.data.suez2018 import load_suez2018
    return load_suez2018()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

DATASETS: Dict[str, DatasetSpec] = {
    "cmipb": DatasetSpec(
        name="cmipb",
        loader=_load_cmipb,
        description="CMI-PB vaccine response: 4 omics assays x 4 timepoints",
        is_longitudinal=True,
        is_multi_omic=True,
        has_groups=True,
        shape_hint="~62 subjects, ~441 variables, 4 timepoints",
        citation="Fourati et al. (2025) PLoS Comp Biol 21(3):e1012927",
    ),
    "suez2018": DatasetSpec(
        name="suez2018",
        loader=_load_suez2018,
        description="Post-antibiotic gut microbiome (16S rRNA, phylum blocks)",
        is_longitudinal=True,
        is_multi_omic=False,
        has_groups=True,
        shape_hint="17 subjects, ~482 taxa, 9 timepoints",
        citation="Suez et al. (2018) Cell 174(6):1406-1423",
    ),
}


def load_dataset(name: str, **kwargs) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Load a dataset by name. Returns ``(X, b, metadata)``."""
    if name not in DATASETS:
        available = ", ".join(sorted(DATASETS.keys()))
        raise ValueError(f"Unknown dataset '{name}'. Available: {available}")
    return DATASETS[name].loader(**kwargs)


def list_datasets() -> Dict[str, DatasetSpec]:
    """Return the full dataset registry."""
    return dict(DATASETS)


def register_dataset(spec: DatasetSpec):
    """Register a new dataset at runtime."""
    DATASETS[spec.name] = spec

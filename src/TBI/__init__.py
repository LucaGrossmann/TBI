"""TBI — Tensor Block Interpreter.

A multilinear extension of Multiple Co-Inertia Analysis for block tensor data,
built on the star-M tensor algebra of Kilmer et al. (2021).
"""

from .TBI_I import TBI_I, TBIResult
from .TBI_II import TBI_II, TBIIResult
from .matrix_MCIA import matrix_MCIA, MCIAResult
from .baselines.block_tpls import block_tpls, BlockTPLSResult
from .result_types import BenchmarkResult, adapt_tbi_i, adapt_tbi_ii, adapt_mcia
from .data import load_dataset, list_datasets

__all__ = [
    "TBI_I", "TBIResult",
    "TBI_II", "TBIIResult",
    "matrix_MCIA", "MCIAResult",
    "block_tpls", "BlockTPLSResult",
    "BenchmarkResult", "adapt_tbi_i", "adapt_tbi_ii", "adapt_mcia",
    "load_dataset", "list_datasets",
]

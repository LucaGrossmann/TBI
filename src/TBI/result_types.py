"""
Unified result types for cross-method benchmarking.

BenchmarkResult is a common container that adapts method-specific result
objects into a comparable format for the benchmark runner.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class BenchmarkResult:
    """Minimal common fields for method comparison."""
    method_name: str
    scores: np.ndarray              # (m, n_components) -- always 2D
    variance_explained: np.ndarray  # (n_components,)
    total_variance: float
    n_iter: int
    elapsed_seconds: float = 0.0


def adapt_tbi_i(result, elapsed: float = 0.0, sheet: int = 0) -> BenchmarkResult:
    """Adapt TBIResult -> BenchmarkResult using scores from one sheet."""
    return BenchmarkResult(
        method_name="TBI-I",
        scores=result.global_scores[:, :result.n_iter, sheet],
        variance_explained=result.variance_explained,
        total_variance=result.total_variance,
        n_iter=result.n_iter,
        elapsed_seconds=elapsed,
    )


def adapt_tbi_ii(result, elapsed: float = 0.0) -> BenchmarkResult:
    """Adapt TBIIResult -> BenchmarkResult."""
    return BenchmarkResult(
        method_name="TBI-II",
        scores=result.global_scores[:, :result.n_iter],
        variance_explained=result.variance_explained,
        total_variance=result.total_variance,
        n_iter=result.n_iter,
        elapsed_seconds=elapsed,
    )


def adapt_mcia(result, elapsed: float = 0.0) -> BenchmarkResult:
    """Adapt MCIAResult -> BenchmarkResult."""
    return BenchmarkResult(
        method_name="Matrix MCIA",
        scores=result.scores,
        variance_explained=result.variance_explained,
        total_variance=result.total_variance,
        n_iter=result.n_iter,
        elapsed_seconds=elapsed,
    )


def adapt_block_tpls(result, elapsed: float = 0.0) -> BenchmarkResult:
    """Adapt BlockTPLSResult -> BenchmarkResult."""
    return BenchmarkResult(
        method_name="block-tPLS",
        scores=result.scores,
        variance_explained=result.variance_explained,
        total_variance=result.total_variance,
        n_iter=result.n_iter,
        elapsed_seconds=elapsed,
    )

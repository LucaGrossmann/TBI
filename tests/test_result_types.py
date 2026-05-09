"""Tests for TBI.result_types — BenchmarkResult adapters for cross-method comparison."""

import numpy as np
import numpy.testing as npt
import pytest
from dataclasses import dataclass
from typing import List

from TBI.result_types import (
    BenchmarkResult,
    adapt_tbi_i,
    adapt_tbi_ii,
    adapt_mcia,
)


# ---------------------------------------------------------------------------
# Mock result dataclasses (mirror the real ones without importing heavy code)
# ---------------------------------------------------------------------------

@dataclass
class MockTBIResult:
    """Mimics TBIResult from TBI_I.py."""
    global_loadings: np.ndarray     # (p, n_iter, n)
    global_scores: np.ndarray       # (m, n_iter, n)
    local_loadings: List[np.ndarray]
    local_scores: List[np.ndarray]
    variance_explained: np.ndarray  # (n_iter,)
    total_variance: float
    n_iter: int


@dataclass
class MockTBIIResult:
    """Mimics TBIIResult from TBI_II.py."""
    global_loadings: np.ndarray      # (p, n_iter)
    global_scores: np.ndarray        # (m, n_iter)
    local_loadings: List[np.ndarray]
    local_scores: List[np.ndarray]
    sheet_indices: np.ndarray        # (n_iter,)
    variance_explained: np.ndarray   # (n_iter,)
    total_variance: float
    n_iter: int


@dataclass
class MockMCIAResult:
    """Mimics MCIAResult from matrix_MCIA.py."""
    scores: np.ndarray              # (m, n_iter)
    loadings: np.ndarray            # (p*n, n_iter)
    variance_explained: np.ndarray  # (n_iter,)
    total_variance: float
    n_iter: int


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(99)


@pytest.fixture
def tbi_i_result(rng):
    m, p, n, n_iter = 10, 6, 3, 4
    return MockTBIResult(
        global_loadings=rng.normal(size=(p, n_iter, n)),
        global_scores=rng.normal(size=(m, n_iter, n)),
        local_loadings=[rng.normal(size=(3, n_iter, n)), rng.normal(size=(3, n_iter, n))],
        local_scores=[rng.normal(size=(m, n_iter, n)), rng.normal(size=(m, n_iter, n))],
        variance_explained=np.array([0.4, 0.2, 0.1, 0.05]),
        total_variance=100.0,
        n_iter=n_iter,
    )


@pytest.fixture
def tbi_ii_result(rng):
    m, p, n_iter = 10, 6, 4
    return MockTBIIResult(
        global_loadings=rng.normal(size=(p, n_iter)),
        global_scores=rng.normal(size=(m, n_iter)),
        local_loadings=[rng.normal(size=(3, n_iter)), rng.normal(size=(3, n_iter))],
        local_scores=[rng.normal(size=(m, n_iter)), rng.normal(size=(m, n_iter))],
        sheet_indices=np.array([0, 1, 0, 2]),
        variance_explained=np.array([0.35, 0.25, 0.15, 0.05]),
        total_variance=80.0,
        n_iter=n_iter,
    )


@pytest.fixture
def mcia_result(rng):
    m, p_times_n, n_iter = 10, 18, 4
    return MockMCIAResult(
        scores=rng.normal(size=(m, n_iter)),
        loadings=rng.normal(size=(p_times_n, n_iter)),
        variance_explained=np.array([0.5, 0.2, 0.1, 0.05]),
        total_variance=120.0,
        n_iter=n_iter,
    )


# ===========================================================================
# adapt_tbi_i
# ===========================================================================

class TestAdaptTbiI:
    """Tests for adapt_tbi_i adapter."""

    def test_returns_benchmark_result(self, tbi_i_result):
        br = adapt_tbi_i(tbi_i_result)
        assert isinstance(br, BenchmarkResult)

    def test_method_name(self, tbi_i_result):
        br = adapt_tbi_i(tbi_i_result)
        assert br.method_name == "TBI-I"

    def test_scores_shape(self, tbi_i_result):
        """Scores should be (m, n_iter) — 2D slice from one sheet."""
        br = adapt_tbi_i(tbi_i_result, sheet=0)
        assert br.scores.ndim == 2
        assert br.scores.shape == (10, 4)

    def test_different_sheet(self, tbi_i_result):
        """Different sheet should produce different scores."""
        br0 = adapt_tbi_i(tbi_i_result, sheet=0)
        br1 = adapt_tbi_i(tbi_i_result, sheet=1)
        assert not np.allclose(br0.scores, br1.scores)

    def test_preserves_variance_explained(self, tbi_i_result):
        br = adapt_tbi_i(tbi_i_result)
        npt.assert_array_equal(br.variance_explained, tbi_i_result.variance_explained)

    def test_preserves_total_variance(self, tbi_i_result):
        br = adapt_tbi_i(tbi_i_result)
        assert br.total_variance == tbi_i_result.total_variance

    def test_preserves_n_iter(self, tbi_i_result):
        br = adapt_tbi_i(tbi_i_result)
        assert br.n_iter == tbi_i_result.n_iter

    def test_elapsed_seconds(self, tbi_i_result):
        br = adapt_tbi_i(tbi_i_result, elapsed=1.5)
        assert br.elapsed_seconds == 1.5

    def test_default_elapsed_zero(self, tbi_i_result):
        br = adapt_tbi_i(tbi_i_result)
        assert br.elapsed_seconds == 0.0


# ===========================================================================
# adapt_tbi_ii
# ===========================================================================

class TestAdaptTbiII:
    """Tests for adapt_tbi_ii adapter."""

    def test_returns_benchmark_result(self, tbi_ii_result):
        br = adapt_tbi_ii(tbi_ii_result)
        assert isinstance(br, BenchmarkResult)

    def test_method_name(self, tbi_ii_result):
        br = adapt_tbi_ii(tbi_ii_result)
        assert br.method_name == "TBI-II"

    def test_scores_shape(self, tbi_ii_result):
        """Scores should be (m, n_iter) — already 2D for TBI-II."""
        br = adapt_tbi_ii(tbi_ii_result)
        assert br.scores.ndim == 2
        assert br.scores.shape == (10, 4)

    def test_preserves_variance_explained(self, tbi_ii_result):
        br = adapt_tbi_ii(tbi_ii_result)
        npt.assert_array_equal(br.variance_explained, tbi_ii_result.variance_explained)

    def test_preserves_total_variance(self, tbi_ii_result):
        br = adapt_tbi_ii(tbi_ii_result)
        assert br.total_variance == tbi_ii_result.total_variance

    def test_preserves_n_iter(self, tbi_ii_result):
        br = adapt_tbi_ii(tbi_ii_result)
        assert br.n_iter == tbi_ii_result.n_iter

    def test_elapsed_seconds(self, tbi_ii_result):
        br = adapt_tbi_ii(tbi_ii_result, elapsed=2.3)
        assert br.elapsed_seconds == 2.3


# ===========================================================================
# adapt_mcia
# ===========================================================================

class TestAdaptMcia:
    """Tests for adapt_mcia adapter."""

    def test_returns_benchmark_result(self, mcia_result):
        br = adapt_mcia(mcia_result)
        assert isinstance(br, BenchmarkResult)

    def test_method_name(self, mcia_result):
        br = adapt_mcia(mcia_result)
        assert br.method_name == "Matrix MCIA"

    def test_scores_shape(self, mcia_result):
        br = adapt_mcia(mcia_result)
        assert br.scores.ndim == 2
        assert br.scores.shape == (10, 4)

    def test_preserves_variance_explained(self, mcia_result):
        br = adapt_mcia(mcia_result)
        npt.assert_array_equal(br.variance_explained, mcia_result.variance_explained)

    def test_preserves_total_variance(self, mcia_result):
        br = adapt_mcia(mcia_result)
        assert br.total_variance == mcia_result.total_variance

    def test_preserves_n_iter(self, mcia_result):
        br = adapt_mcia(mcia_result)
        assert br.n_iter == mcia_result.n_iter

    def test_elapsed_seconds(self, mcia_result):
        br = adapt_mcia(mcia_result, elapsed=0.7)
        assert br.elapsed_seconds == 0.7


# ===========================================================================
# BenchmarkResult dataclass
# ===========================================================================

class TestBenchmarkResult:
    """Basic sanity checks for the BenchmarkResult dataclass."""

    def test_fields(self):
        br = BenchmarkResult(
            method_name="test",
            scores=np.zeros((5, 2)),
            variance_explained=np.array([0.5, 0.3]),
            total_variance=10.0,
            n_iter=2,
        )
        assert br.method_name == "test"
        assert br.scores.shape == (5, 2)
        assert br.elapsed_seconds == 0.0  # default

    def test_custom_elapsed(self):
        br = BenchmarkResult(
            method_name="x",
            scores=np.zeros((3, 1)),
            variance_explained=np.array([1.0]),
            total_variance=5.0,
            n_iter=1,
            elapsed_seconds=3.14,
        )
        assert br.elapsed_seconds == 3.14

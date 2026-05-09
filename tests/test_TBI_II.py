"""Tests for TBI Algorithm II (greedy)."""

import time
import numpy as np
import numpy.testing as npt
import pytest

from TBI.TBI_II import (
    TBI_II, TBIIResult,
    _svd_all_sheets, _update_svd_cache, _find_best_sheet,
    _compute_scores_single,
    _compute_block_loadings_single, _compute_block_scores_single,
    _deflate_single,
)
from TBI.helpers import _block_ranges, _compute_total_variance, randorth
from TBI.normalization import default_normalize, mcia_normalize, no_normalize


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def timer(request):
    """Print wall-clock time for every test."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"  [{request.node.name}] {elapsed:.6f}s")


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def simple_tensor(rng):
    """(m=8, p=6, n=3) tensor with b=[0,2,4] (3 blocks of size 2)."""
    m, p, n = 8, 6, 3
    X = rng.normal(size=(m, p, n))
    b = np.array([0, 2, 4])
    M = np.eye(n)
    return X, b, M


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def test_smoke_returns_TBIIResult(simple_tensor):
    X, b, M = simple_tensor
    result = TBI_II(X, b, M, energy=0.99, max_iter=3)
    assert isinstance(result, TBIIResult)


# ---------------------------------------------------------------------------
# Output shapes — 2D (not 3D like TBI-I)
# ---------------------------------------------------------------------------

def test_output_shapes(simple_tensor):
    X, b, M = simple_tensor
    m, p, n = X.shape
    result = TBI_II(X, b, M, energy=0.99, max_iter=4)
    ni = result.n_iter

    assert result.global_loadings.shape == (p, ni)
    assert result.global_scores.shape == (m, ni)

    assert len(result.local_loadings) == b.shape[0]
    assert len(result.local_scores) == b.shape[0]

    # Block sizes: [0:2], [2:4], [4:6]
    for idx, (_, start, end) in enumerate(_block_ranges(b, p)):
        p_k = end - start
        assert result.local_loadings[idx].shape == (p_k, ni)
        assert result.local_scores[idx].shape == (m, ni)

    assert result.sheet_indices.shape == (ni,)
    assert result.sheet_indices.dtype == int
    assert result.variance_explained.shape == (ni,)
    assert result.total_variance > 0


# ---------------------------------------------------------------------------
# Sheet indices are valid
# ---------------------------------------------------------------------------

def test_sheet_indices_in_range(simple_tensor):
    X, b, M = simple_tensor
    n = X.shape[2]
    result = TBI_II(X, b, M, energy=0.99, max_iter=5)
    assert np.all(result.sheet_indices >= 0)
    assert np.all(result.sheet_indices < n)


# ---------------------------------------------------------------------------
# Energy / variance explained
# ---------------------------------------------------------------------------

def test_variance_explained_is_nonnegative(simple_tensor):
    X, b, M = simple_tensor
    result = TBI_II(X, b, M, energy=0.99, max_iter=5)
    assert np.all(result.variance_explained >= -1e-10)


def test_energy_monotonically_increases(simple_tensor):
    X, b, M = simple_tensor
    result = TBI_II(X, b, M, energy=0.99, max_iter=5)
    cumulative = np.cumsum(result.variance_explained)
    assert np.all(np.diff(cumulative) >= -1e-10)


def test_cumulative_energy_does_not_exceed_total(simple_tensor):
    X, b, M = simple_tensor
    result = TBI_II(X, b, M, energy=0.99, max_iter=5)
    total_explained = np.sum(result.variance_explained)
    assert total_explained <= result.total_variance + 1e-10


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

def test_early_stopping_low_energy(simple_tensor):
    """With a very low energy threshold, should stop after few iterations."""
    X, b, M = simple_tensor
    result = TBI_II(X, b, M, energy=0.01, max_iter=10)
    assert result.n_iter < 10


def test_max_iter_respected(simple_tensor):
    X, b, M = simple_tensor
    result = TBI_II(X, b, M, energy=1.0, max_iter=3)
    assert result.n_iter <= 3


# ---------------------------------------------------------------------------
# Deflation reduces energy
# ---------------------------------------------------------------------------

def test_deflation_reduces_variance(rng):
    m, p, n = 10, 8, 4
    X = rng.normal(size=(m, p, n))
    b = np.array([0, 3, 6])
    M = np.eye(n)

    result = TBI_II(X, b, M, energy=0.99, max_iter=5, normalize_fn=no_normalize)
    assert np.sum(result.variance_explained) > 0


# ---------------------------------------------------------------------------
# Custom normalization
# ---------------------------------------------------------------------------

def test_custom_normalize_fn(rng):
    m, p, n = 10, 6, 3
    X = rng.normal(size=(m, p, n))
    b = np.array([0, 3])
    M = np.eye(n)

    from TBI.normalization import variable_mean_center
    def my_norm(X_hat, b):
        return variable_mean_center(X_hat.copy(), sheet_level=False)

    result = TBI_II(X, b, M, energy=0.99, max_iter=3, normalize_fn=my_norm)
    assert isinstance(result, TBIIResult)
    assert result.n_iter > 0


def test_no_normalize(rng):
    m, p, n = 10, 6, 3
    X = rng.normal(size=(m, p, n))
    b = np.array([0, 3])
    M = np.eye(n)

    result = TBI_II(X, b, M, energy=0.99, max_iter=3, normalize_fn=no_normalize)
    assert isinstance(result, TBIIResult)


def test_mcia_normalize(rng):
    m, p, n = 10, 6, 3
    X = rng.uniform(1, 10, size=(m, p, n))  # positive values for MCIA
    b = np.array([0, 3])
    M = np.eye(n)

    result = TBI_II(X, b, M, energy=0.99, max_iter=3, normalize_fn=mcia_normalize)
    assert isinstance(result, TBIIResult)


# ---------------------------------------------------------------------------
# Invalid input raises ValueError
# ---------------------------------------------------------------------------

def test_invalid_X_ndim():
    with pytest.raises(ValueError):
        TBI_II(np.zeros((3, 4)), np.array([0, 2]), np.eye(1))


def test_invalid_M_not_square():
    with pytest.raises(ValueError):
        TBI_II(np.zeros((3, 4, 2)), np.array([0, 2]), np.zeros((2, 3)))


def test_invalid_M_wrong_size():
    with pytest.raises(ValueError):
        TBI_II(np.zeros((3, 4, 2)), np.array([0, 2]), np.eye(5))


def test_invalid_b_not_1d():
    with pytest.raises(ValueError):
        TBI_II(np.zeros((3, 4, 2)), np.array([[0, 2]]), np.eye(2))


def test_invalid_b_out_of_range():
    with pytest.raises(ValueError):
        TBI_II(np.zeros((3, 4, 2)), np.array([0, 10]), np.eye(2))


# ---------------------------------------------------------------------------
# Rank-1 known tensor
# ---------------------------------------------------------------------------

def test_rank1_tensor_captured_in_one_iteration(rng):
    """A rank-1 tensor (per sheet) should be almost fully captured in 1 iter."""
    m, p, n = 10, 6, 3
    b = np.array([0, 3])

    X = np.zeros((m, p, n))
    for i in range(n):
        u = rng.normal(size=(m, 1))
        v = rng.normal(size=(p, 1))
        X[:, :, i] = u @ v.T

    M = np.eye(n)
    result = TBI_II(X, b, M, energy=0.99, max_iter=5, normalize_fn=no_normalize)

    # First iteration should capture a large fraction of one sheet
    assert result.variance_explained[0] / result.total_variance > 0.20


# ---------------------------------------------------------------------------
# Non-identity M
# ---------------------------------------------------------------------------

def test_with_random_orthogonal_M(rng):
    m, p, n = 10, 8, 4
    X = rng.normal(size=(m, p, n))
    b = np.array([0, 3, 6])
    M = randorth(n)

    result = TBI_II(X, b, M, energy=0.99, max_iter=4)
    assert isinstance(result, TBIIResult)
    assert result.n_iter > 0
    assert result.total_variance > 0


# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------

def test_find_best_sheet_returns_correct_sheet(rng):
    """Put all variance in sheet 2, check it gets picked."""
    m, p, n = 8, 6, 3
    X = np.zeros((m, p, n))
    X[:, :, 2] = rng.normal(size=(m, p)) * 10  # sheet 2 dominant

    cache = _svd_all_sheets(X)
    loading, sheet_idx, sigma_sq = _find_best_sheet(cache)
    assert sheet_idx == 2
    assert sigma_sq > 0
    npt.assert_allclose(np.linalg.norm(loading), 1.0, atol=1e-10)


def test_find_best_sheet_loading_unit_norm(rng):
    X = rng.normal(size=(10, 6, 4))
    cache = _svd_all_sheets(X)
    loading, _, _ = _find_best_sheet(cache)
    npt.assert_allclose(np.linalg.norm(loading), 1.0, atol=1e-10)


def test_update_svd_cache(rng):
    """After deflation, updating the cache should reflect the new SVD."""
    m, p, n = 10, 6, 3
    X = rng.normal(size=(m, p, n))
    cache = _svd_all_sheets(X)

    # Deflate sheet 1
    b = np.array([0, 3])
    scores = X[:, :, 1] @ cache[1][1]
    bl = _compute_block_loadings_single(X, b, scores, 1)
    _deflate_single(X, b, bl, 1)

    # Cache is now stale for sheet 1
    _update_svd_cache(X, cache, 1)

    # Recompute from scratch and compare
    fresh_cache = _svd_all_sheets(X)
    npt.assert_allclose(cache[1][0], fresh_cache[1][0], atol=1e-10)
    npt.assert_allclose(np.abs(cache[1][1]), np.abs(fresh_cache[1][1]), atol=1e-10)


def test_deflate_single_reduces_sheet_variance(rng):
    m, p, n = 10, 6, 3
    X = rng.normal(size=(m, p, n))
    b = np.array([0, 3])
    sheet_idx = 1

    var_before = np.sum(X[:, :, sheet_idx] ** 2)

    loading = np.linalg.svd(X[:, :, sheet_idx], full_matrices=False)[2][0]
    scores = X[:, :, sheet_idx] @ loading
    block_loadings = _compute_block_loadings_single(X, b, scores, sheet_idx)
    _deflate_single(X, b, block_loadings, sheet_idx)

    var_after = np.sum(X[:, :, sheet_idx] ** 2)
    assert var_after < var_before


def test_block_loadings_single_unit_norm(rng):
    m, p, n = 10, 8, 3
    X = rng.normal(size=(m, p, n))
    b = np.array([0, 3, 6])
    scores = rng.normal(size=(m,))

    block_loadings = _compute_block_loadings_single(X, b, scores, sheet_idx=1)
    for bl in block_loadings:
        norm = np.linalg.norm(bl)
        assert norm < 1e-10 or abs(norm - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# Single-block tests (b=[0])
# ---------------------------------------------------------------------------

def test_single_block_smoke(rng):
    """TBI-II should work with a single block (equivalent to greedy tensor PCA)."""
    m, p, n = 10, 6, 3
    X = rng.normal(size=(m, p, n))
    b = np.array([0])
    M = np.eye(n)
    result = TBI_II(X, b, M, energy=0.95, max_iter=10)

    assert isinstance(result, TBIIResult)
    assert result.n_iter >= 1
    assert result.global_loadings.shape == (p, result.n_iter)
    assert result.global_scores.shape == (m, result.n_iter)
    assert len(result.local_loadings) == 1
    assert len(result.local_scores) == 1
    assert result.local_loadings[0].shape == (p, result.n_iter)
    assert result.sheet_indices.shape == (result.n_iter,)
    assert result.variance_explained.sum() > 0


def test_single_block_energy(rng):
    """Single-block TBI-II should capture variance and stop at energy threshold."""
    m, p, n = 15, 8, 3
    X = rng.normal(size=(m, p, n))
    b = np.array([0])
    M = np.eye(n)
    result = TBI_II(X, b, M, energy=0.50, max_iter=30)

    cum_energy = result.variance_explained.sum() / result.total_variance
    assert cum_energy >= 0.50

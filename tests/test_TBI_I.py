"""Tests for TBI Algorithm I."""

import numpy as np
import numpy.testing as npt
import pytest

from TBI.TBI_I import (
    TBI_I, TBIResult,
    _compute_global_loadings, _compute_global_scores,
    _compute_block_loadings, _compute_block_scores, _deflate,
)
from TBI.helpers import _block_ranges, _compute_total_variance, randorth
from TBI.normalization import default_normalize, mcia_normalize, no_normalize


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

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

def test_smoke_returns_TBIResult(simple_tensor):
    X, b, M = simple_tensor
    result = TBI_I(X, b, M, energy=0.99, max_iter=3)
    assert isinstance(result, TBIResult)


# ---------------------------------------------------------------------------
# Output shapes
# ---------------------------------------------------------------------------

def test_output_shapes(simple_tensor):
    X, b, M = simple_tensor
    m, p, n = X.shape
    result = TBI_I(X, b, M, energy=0.99, max_iter=4)
    ni = result.n_iter

    assert result.global_loadings.shape == (p, ni, n)
    assert result.global_scores.shape == (m, ni, n)

    assert len(result.local_loadings) == b.shape[0]
    assert len(result.local_scores) == b.shape[0]

    # Block sizes: [0:2], [2:4], [4:6]
    for idx, (_, start, end) in enumerate(_block_ranges(b, p)):
        p_k = end - start
        assert result.local_loadings[idx].shape == (p_k, ni, n)
        assert result.local_scores[idx].shape == (m, ni, n)

    assert result.variance_explained.shape == (ni,)
    assert result.total_variance > 0


# ---------------------------------------------------------------------------
# Energy / variance explained
# ---------------------------------------------------------------------------

def test_variance_explained_is_nonnegative(simple_tensor):
    X, b, M = simple_tensor
    result = TBI_I(X, b, M, energy=0.99, max_iter=5)
    assert np.all(result.variance_explained >= -1e-10)


def test_energy_monotonically_increases(simple_tensor):
    X, b, M = simple_tensor
    result = TBI_I(X, b, M, energy=0.99, max_iter=5)
    cumulative = np.cumsum(result.variance_explained)
    # Each cumulative entry should be >= the previous
    assert np.all(np.diff(cumulative) >= -1e-10)


def test_cumulative_energy_does_not_exceed_total(simple_tensor):
    X, b, M = simple_tensor
    result = TBI_I(X, b, M, energy=0.99, max_iter=5)
    total_explained = np.sum(result.variance_explained)
    assert total_explained <= result.total_variance + 1e-10


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

def test_early_stopping_low_energy(simple_tensor):
    """With a very low energy threshold, should stop after 1 iteration."""
    X, b, M = simple_tensor
    result = TBI_I(X, b, M, energy=0.01, max_iter=10)
    assert result.n_iter < 10


def test_max_iter_respected(simple_tensor):
    X, b, M = simple_tensor
    result = TBI_I(X, b, M, energy=1.0, max_iter=3)
    assert result.n_iter <= 3


# ---------------------------------------------------------------------------
# Deflation reduces energy
# ---------------------------------------------------------------------------

def test_deflation_reduces_variance(rng):
    m, p, n = 10, 8, 4
    X = rng.normal(size=(m, p, n))
    b = np.array([0, 3, 6])
    M = np.eye(n)

    result = TBI_I(X, b, M, energy=0.99, max_iter=5, normalize_fn=no_normalize)
    # After extracting components, remaining variance should be less
    assert np.sum(result.variance_explained) > 0


# ---------------------------------------------------------------------------
# Custom normalization
# ---------------------------------------------------------------------------

def test_custom_normalize_fn(rng):
    m, p, n = 10, 6, 3
    X = rng.normal(size=(m, p, n))
    b = np.array([0, 3])
    M = np.eye(n)

    # Custom: just mean-center (no block norm)
    from TBI.normalization import variable_mean_center
    def my_norm(X_hat, b):
        return variable_mean_center(X_hat.copy(), sheet_level=False)

    result = TBI_I(X, b, M, energy=0.99, max_iter=3, normalize_fn=my_norm)
    assert isinstance(result, TBIResult)
    assert result.n_iter > 0


def test_no_normalize(rng):
    m, p, n = 10, 6, 3
    X = rng.normal(size=(m, p, n))
    b = np.array([0, 3])
    M = np.eye(n)

    result = TBI_I(X, b, M, energy=0.99, max_iter=3, normalize_fn=no_normalize)
    assert isinstance(result, TBIResult)


def test_mcia_normalize(rng):
    m, p, n = 10, 6, 3
    X = rng.uniform(1, 10, size=(m, p, n))  # positive values for MCIA
    b = np.array([0, 3])
    M = np.eye(n)

    result = TBI_I(X, b, M, energy=0.99, max_iter=3, normalize_fn=mcia_normalize)
    assert isinstance(result, TBIResult)


# ---------------------------------------------------------------------------
# Invalid input raises ValueError
# ---------------------------------------------------------------------------

def test_invalid_X_ndim():
    with pytest.raises(ValueError):
        TBI_I(np.zeros((3, 4)), np.array([0, 2]), np.eye(1))


def test_invalid_M_not_square():
    with pytest.raises(ValueError):
        TBI_I(np.zeros((3, 4, 2)), np.array([0, 2]), np.zeros((2, 3)))


def test_invalid_M_wrong_size():
    with pytest.raises(ValueError):
        TBI_I(np.zeros((3, 4, 2)), np.array([0, 2]), np.eye(5))


def test_invalid_b_not_1d():
    with pytest.raises(ValueError):
        TBI_I(np.zeros((3, 4, 2)), np.array([[0, 2]]), np.eye(2))


def test_invalid_b_out_of_range():
    with pytest.raises(ValueError):
        TBI_I(np.zeros((3, 4, 2)), np.array([0, 10]), np.eye(2))


# ---------------------------------------------------------------------------
# Rank-1 known tensor
# ---------------------------------------------------------------------------

def test_rank1_tensor_captured_in_one_iteration(rng):
    """A rank-1 tensor (per sheet) should be almost fully captured in 1 iteration."""
    m, p, n = 10, 6, 3
    b = np.array([0, 3])

    # Build rank-1 tensor per sheet: X[:,:,i] = u_i @ v_i.T
    X = np.zeros((m, p, n))
    for i in range(n):
        u = rng.normal(size=(m, 1))
        v = rng.normal(size=(p, 1))
        X[:, :, i] = u @ v.T

    M = np.eye(n)
    result = TBI_I(X, b, M, energy=0.99, max_iter=5, normalize_fn=no_normalize)

    # First iteration should capture nearly all variance
    ratio = result.variance_explained[0] / result.total_variance
    assert ratio > 0.90


# ---------------------------------------------------------------------------
# Non-identity M
# ---------------------------------------------------------------------------

def test_with_random_orthogonal_M(rng):
    m, p, n = 10, 8, 4
    X = rng.normal(size=(m, p, n))
    b = np.array([0, 3, 6])
    M = randorth(n)

    result = TBI_I(X, b, M, energy=0.99, max_iter=4)
    assert isinstance(result, TBIResult)
    assert result.n_iter > 0
    assert result.total_variance > 0


# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------

def test_block_ranges():
    b = np.array([0, 3, 7])
    p = 10
    ranges = list(_block_ranges(b, p))
    assert ranges == [(0, 0, 3), (1, 3, 7), (2, 7, 10)]


def test_compute_total_variance(rng):
    X = rng.normal(size=(4, 3, 2))
    var = _compute_total_variance(X)
    expected = np.sum(X ** 2)
    npt.assert_allclose(var, expected, atol=1e-12)


def test_global_loadings_unit_norm(rng):
    X = rng.normal(size=(10, 6, 3))
    loadings = _compute_global_loadings(X)
    for i in range(X.shape[2]):
        npt.assert_allclose(np.linalg.norm(loadings[:, i]), 1.0, atol=1e-10)


def test_block_loadings_unit_norm(rng):
    m, p, n = 10, 8, 3
    X = rng.normal(size=(m, p, n))
    b = np.array([0, 3, 6])
    scores = rng.normal(size=(m, n))

    block_loadings = _compute_block_loadings(X, b, scores)
    for bl in block_loadings:
        for i in range(n):
            norm = np.linalg.norm(bl[:, i])
            # Should be 1 or 0 (if numerator was zero)
            assert norm < 1e-10 or abs(norm - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# Single-block tests (b=[0])
# ---------------------------------------------------------------------------

def test_single_block_smoke(rng):
    """TBI-I should work with a single block (equivalent to tensor PCA)."""
    m, p, n = 10, 6, 3
    X = rng.normal(size=(m, p, n))
    b = np.array([0])
    M = np.eye(n)
    result = TBI_I(X, b, M, energy=0.95, max_iter=5)

    assert isinstance(result, TBIResult)
    assert result.n_iter >= 1
    assert result.global_loadings.shape == (p, result.n_iter, n)
    assert result.global_scores.shape == (m, result.n_iter, n)
    assert len(result.local_loadings) == 1
    assert len(result.local_scores) == 1
    assert result.local_loadings[0].shape == (p, result.n_iter, n)
    assert result.variance_explained.sum() > 0


def test_single_block_energy(rng):
    """Single-block TBI-I should capture variance and stop at energy threshold."""
    m, p, n = 15, 8, 3
    X = rng.normal(size=(m, p, n))
    b = np.array([0])
    M = np.eye(n)
    result = TBI_I(X, b, M, energy=0.50, max_iter=20)

    cum_energy = result.variance_explained.sum() / result.total_variance
    assert cum_energy >= 0.50

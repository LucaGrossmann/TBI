import numpy as np
import numpy.testing as npt
import pytest
from TBI.matrix_MCIA import matrix_MCIA, MCIAResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_tensor():
    """Random (8, 10, 4) tensor with 2 blocks."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(8, 10, 4))
    b = np.array([0, 5])
    return X, b


@pytest.fixture
def dct_matrix_4():
    """Orthonormal 4x4 DCT matrix."""
    N = 4
    i = np.arange(N).reshape((N, 1))
    j = np.arange(N)
    D = np.sqrt(2.0 / N) * np.cos(np.pi * (2 * j + 1) * i / (2 * N))
    D[0, :] = np.sqrt(1.0 / N)
    return D


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

def test_smoke_returns_MCIAResult(small_tensor):
    X, b = small_tensor
    result = matrix_MCIA(X, b)
    assert isinstance(result, MCIAResult)


def test_output_shapes(small_tensor):
    X, b = small_tensor
    m, p, n = X.shape
    result = matrix_MCIA(X, b, energy=0.99, max_iter=5)

    assert result.scores.shape == (m, result.n_iter)
    assert result.loadings.shape == (p * n, result.n_iter)
    assert result.variance_explained.shape == (result.n_iter,)
    assert result.n_iter <= 5


# ---------------------------------------------------------------------------
# Variance properties
# ---------------------------------------------------------------------------

def test_variance_explained_is_nonnegative(small_tensor):
    X, b = small_tensor
    result = matrix_MCIA(X, b, max_iter=5)
    assert np.all(result.variance_explained >= 0)


def test_energy_monotonically_increases(small_tensor):
    X, b = small_tensor
    result = matrix_MCIA(X, b, energy=0.99, max_iter=7)
    cumulative = np.cumsum(result.variance_explained)
    assert np.all(np.diff(cumulative) >= 0)


def test_cumulative_energy_does_not_exceed_total(small_tensor):
    X, b = small_tensor
    result = matrix_MCIA(X, b, energy=0.999, max_iter=8)
    cum = result.variance_explained.sum()
    assert cum <= result.total_variance + 1e-8


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

def test_early_stopping_low_energy(small_tensor):
    X, b = small_tensor
    result_low = matrix_MCIA(X, b, energy=0.30, max_iter=8)
    result_high = matrix_MCIA(X, b, energy=0.99, max_iter=8)
    assert result_low.n_iter <= result_high.n_iter


def test_max_iter_respected(small_tensor):
    X, b = small_tensor
    result = matrix_MCIA(X, b, energy=0.999, max_iter=3)
    assert result.n_iter <= 3


# ---------------------------------------------------------------------------
# Deflation
# ---------------------------------------------------------------------------

def test_deflation_reduces_variance(small_tensor):
    """Variance explained should be non-increasing (deflation removes largest component first)."""
    X, b = small_tensor
    result = matrix_MCIA(X, b, energy=0.999, max_iter=5)
    if result.n_iter > 1:
        assert np.all(np.diff(result.variance_explained) <= 1e-10)


# ---------------------------------------------------------------------------
# DCT transform
# ---------------------------------------------------------------------------

def test_with_dct_transform(dct_matrix_4):
    rng = np.random.default_rng(7)
    X = rng.normal(size=(10, 12, 4))
    b = np.array([0, 4, 8])
    result = matrix_MCIA(X, b, energy=0.95, max_iter=8)

    assert isinstance(result, MCIAResult)
    assert result.n_iter > 0
    cum = result.variance_explained.sum() / result.total_variance
    assert cum >= 0.85  # should capture substantial variance


# ---------------------------------------------------------------------------
# Invalid inputs
# ---------------------------------------------------------------------------

def test_invalid_X_ndim():
    X = np.zeros((3, 4))
    b = np.array([0, 2])
    with pytest.raises((ValueError, AttributeError)):
        matrix_MCIA(X, b)


# ---------------------------------------------------------------------------
# Single block
# ---------------------------------------------------------------------------

def test_single_block_smoke():
    rng = np.random.default_rng(99)
    X = rng.normal(size=(8, 6, 3))
    b = np.array([0])
    result = matrix_MCIA(X, b, energy=0.95, max_iter=5)
    assert result.n_iter > 0
    assert result.scores.shape[0] == 8


# ---------------------------------------------------------------------------
# Rank-1 tensor
# ---------------------------------------------------------------------------

def test_rank1_tensor_captured_in_one_iteration():
    """A rank-1 tensor (one component) should be captured in 1 iteration."""
    rng = np.random.default_rng(12)
    m, p, n = 10, 6, 3
    u = rng.normal(size=m)
    v = rng.normal(size=p)
    w = rng.normal(size=n)
    X = np.einsum("i,j,k->ijk", u, v, w)
    b = np.array([0, 3])

    result = matrix_MCIA(X, b, energy=0.99, max_iter=5)
    # First component should capture nearly all variance
    ratio = result.variance_explained[0] / result.total_variance
    assert ratio > 0.99


# ---------------------------------------------------------------------------
# Scores are orthogonal
# ---------------------------------------------------------------------------

def test_scores_are_orthogonal(small_tensor):
    """SVD-based deflation should produce orthogonal score vectors."""
    X, b = small_tensor
    result = matrix_MCIA(X, b, energy=0.99, max_iter=5)
    if result.n_iter >= 2:
        gram = result.scores.T @ result.scores
        off_diag = gram - np.diag(np.diag(gram))
        npt.assert_allclose(off_diag, 0.0, atol=1e-8)

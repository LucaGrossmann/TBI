"""Tests for block.tPLS (tensorOmics block Tensor PLS)."""

import numpy as np
import numpy.testing as npt
import pytest

from TBI.baselines.block_tpls import (
    block_tpls, BlockTPLSResult,
    _split_blocks, _mdf_center, _rgcca_nipals_slice,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_tensor():
    """Small (10, 8, 3) tensor with 2 blocks for smoke tests."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((10, 8, 3))
    b = np.array([0, 4])
    return X, b


@pytest.fixture
def three_block_tensor():
    """(15, 12, 4) tensor with 3 blocks."""
    rng = np.random.default_rng(99)
    X = rng.standard_normal((15, 12, 4))
    b = np.array([0, 4, 8])
    return X, b


@pytest.fixture
def known_signal_tensor():
    """Tensor with a known shared latent factor across blocks.

    A single latent score vector drives all blocks, so block.tPLS
    should recover it with high alignment.
    """
    rng = np.random.default_rng(77)
    m, n = 20, 3

    # Shared latent score (one per subject)
    latent = rng.standard_normal(m)
    latent -= latent.mean()

    # Block 1: 6 variables, latent drives first variable strongly
    p1 = 6
    B1 = rng.standard_normal((m, p1, n)) * 0.1
    loading1 = rng.standard_normal(p1)
    loading1 /= np.linalg.norm(loading1)
    for k in range(n):
        B1[:, :, k] += 5.0 * np.outer(latent, loading1)

    # Block 2: 5 variables, same latent drives it
    p2 = 5
    B2 = rng.standard_normal((m, p2, n)) * 0.1
    loading2 = rng.standard_normal(p2)
    loading2 /= np.linalg.norm(loading2)
    for k in range(n):
        B2[:, :, k] += 5.0 * np.outer(latent, loading2)

    X = np.concatenate([B1, B2], axis=1)
    b = np.array([0, p1])
    return X, b, latent, loading1, loading2


# ---------------------------------------------------------------------------
# Smoke and shape tests
# ---------------------------------------------------------------------------

class TestSmoke:
    def test_returns_result_type(self, small_tensor):
        X, b = small_tensor
        result = block_tpls(X, b, n_components=3)
        assert isinstance(result, BlockTPLSResult)

    def test_output_shapes(self, small_tensor):
        X, b = small_tensor
        m, p, n = X.shape
        Q = len(b)
        result = block_tpls(X, b, n_components=3)

        assert result.scores.shape[0] == m
        assert result.scores.shape[1] == result.n_iter
        assert len(result.block_scores) == Q
        assert len(result.loadings) == Q
        for q in range(Q):
            assert result.block_scores[q].shape == (m, result.n_iter)
        assert len(result.variance_explained) == result.n_iter
        assert len(result.sheet_indices) == result.n_iter

    def test_three_blocks(self, three_block_tensor):
        X, b = three_block_tensor
        result = block_tpls(X, b, n_components=3)
        assert isinstance(result, BlockTPLSResult)
        assert len(result.block_scores) == 3
        assert len(result.loadings) == 3

    def test_single_component(self, small_tensor):
        X, b = small_tensor
        result = block_tpls(X, b, n_components=1)
        assert result.n_iter == 1


# ---------------------------------------------------------------------------
# Variance properties
# ---------------------------------------------------------------------------

class TestVariance:
    def test_variance_non_negative(self, small_tensor):
        X, b = small_tensor
        result = block_tpls(X, b, n_components=5)
        assert np.all(result.variance_explained >= 0)
        assert result.total_variance >= 0

    def test_variance_bounded(self, small_tensor):
        X, b = small_tensor
        result = block_tpls(X, b, n_components=20)
        assert result.variance_explained.sum() <= result.total_variance + 1e-8

    def test_deflation_reduces_residual(self, small_tensor):
        """Each component should capture some variance (early components at least)."""
        X, b = small_tensor
        result = block_tpls(X, b, n_components=3)
        # First component should capture the most
        assert result.variance_explained[0] > 0


# ---------------------------------------------------------------------------
# Loading properties
# ---------------------------------------------------------------------------

class TestLoadings:
    def test_loading_unit_norm(self, small_tensor):
        X, b = small_tensor
        result = block_tpls(X, b, n_components=3)
        for q in range(len(b)):
            for h in range(result.n_iter):
                norm = np.linalg.norm(result.loadings[q][:, h])
                npt.assert_allclose(norm, 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# MDF centering
# ---------------------------------------------------------------------------

class TestMDF:
    def test_mdf_center_subtracts_mean(self):
        rng = np.random.default_rng(10)
        X = rng.standard_normal((8, 5, 3))
        X_centered = _mdf_center(X)

        # Mean across subjects should be ~zero for each (var, time)
        subject_mean = X_centered.mean(axis=0)
        npt.assert_allclose(subject_mean, 0.0, atol=1e-12)

    def test_no_mdf_option(self, small_tensor):
        """use_mdf=False should skip centering."""
        X, b = small_tensor
        result = block_tpls(X, b, use_mdf=False, n_components=2)
        assert isinstance(result, BlockTPLSResult)


# ---------------------------------------------------------------------------
# Design matrix
# ---------------------------------------------------------------------------

class TestDesignMatrix:
    def test_custom_design_matrix(self, three_block_tensor):
        """Custom C connecting only adjacent blocks."""
        X, b = three_block_tensor
        C = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ], dtype=float)
        result = block_tpls(X, b, C=C, n_components=3)
        assert isinstance(result, BlockTPLSResult)
        assert result.n_iter > 0


# ---------------------------------------------------------------------------
# Known signal recovery
# ---------------------------------------------------------------------------

class TestSignalRecovery:
    def test_recovers_shared_latent(self, known_signal_tensor):
        """block.tPLS should recover a shared latent factor across blocks."""
        X, b, latent, loading1, loading2 = known_signal_tensor
        result = block_tpls(X, b, n_components=3)

        # Global scores first component should align with the latent factor
        s = result.scores[:, 0]
        alignment = abs(np.dot(s, latent)) / (np.linalg.norm(s) * np.linalg.norm(latent))
        assert alignment > 0.8, f"Score-latent alignment: {alignment:.3f}"

    def test_block_loadings_align(self, known_signal_tensor):
        """Recovered loadings should align with true loadings."""
        X, b, latent, loading1, loading2 = known_signal_tensor
        result = block_tpls(X, b, n_components=3)

        # Block loadings should align with true loadings (sign-agnostic)
        a1 = result.loadings[0][:, 0]
        a2 = result.loadings[1][:, 0]
        align1 = abs(np.dot(a1, loading1))
        align2 = abs(np.dot(a2, loading2))
        assert align1 > 0.8, f"Block 1 loading alignment: {align1:.3f}"
        assert align2 > 0.8, f"Block 2 loading alignment: {align2:.3f}"


# ---------------------------------------------------------------------------
# Helper tests
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_split_blocks(self):
        X = np.arange(60).reshape(3, 10, 2)
        b = np.array([0, 4, 7])
        blocks = _split_blocks(X, b)
        assert len(blocks) == 3
        assert blocks[0].shape == (3, 4, 2)
        assert blocks[1].shape == (3, 3, 2)
        assert blocks[2].shape == (3, 3, 2)

    def test_rgcca_nipals_converges(self):
        """RGCCA NIPALS should converge on a small problem."""
        rng = np.random.default_rng(42)
        blocks = [rng.standard_normal((10, 5)), rng.standard_normal((10, 4))]
        C = np.array([[0, 1], [1, 0]], dtype=float)
        loadings, criterion, n_inner = _rgcca_nipals_slice(blocks, C, tol=1e-8)
        assert n_inner < 100  # should converge well before max
        assert criterion > 0
        npt.assert_allclose(np.linalg.norm(loadings[0]), 1.0, atol=1e-10)
        npt.assert_allclose(np.linalg.norm(loadings[1]), 1.0, atol=1e-10)

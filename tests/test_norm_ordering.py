"""
Tests for normalization ordering: pre-transform vs post-transform.

Verifies that both orderings produce valid (finite, non-NaN) results
and that the post-transform ordering (TBI default) captures positive variance.

Tests for normalization ordering effects on TBI.
"""

import numpy as np
import numpy.testing as npt
import pytest

from TBI import TBI_I
from TBI.star_M import mode3
from TBI.normalization import default_normalize
from TBI.analysis_utils import dct_matrix
from TBI.helpers import _compute_total_variance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_block_tensor(m=30, blocks=(15, 20, 10), n=4, rng=None):
    """Small synthetic block tensor with different scales per block."""
    if rng is None:
        rng = np.random.default_rng(42)

    p = sum(blocks)
    X = np.zeros((m, p, n))
    t = np.linspace(0, 2 * np.pi, n)
    subject_weights = rng.standard_normal(m)

    col = 0
    for k, pk in enumerate(blocks):
        scale = [1.0, 100.0, 10000.0][k]
        temporal = np.sin((k + 1) * t)
        noise = rng.standard_normal((m, pk, n)) * scale * 0.3
        block_loadings = rng.standard_normal((pk,))

        for i in range(n):
            X[:, col:col + pk, i] = (
                scale * np.outer(subject_weights * temporal[i], block_loadings)
                + noise[:, :, i]
            )
        col += pk

    b = np.array([0, blocks[0], blocks[0] + blocks[1]])
    return X, b


def _no_normalize(X_hat, b):
    """Identity normalization -- pass-through."""
    return X_hat


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestNormOrderingPostTransform:
    """Post-transform normalization (TBI default): mode3 -> normalize -> decompose."""

    def test_post_transform_results_finite(self):
        """Post-transform normalization should produce finite, non-NaN results."""
        X, b = _make_block_tensor()
        M = dct_matrix(X.shape[2])
        result = TBI_I(X, b, M, energy=0.9, max_iter=5)

        assert np.all(np.isfinite(result.global_scores))
        assert np.all(np.isfinite(result.global_loadings))
        assert np.all(np.isfinite(result.variance_explained))
        assert not np.any(np.isnan(result.variance_explained))

    def test_post_transform_captures_positive_variance(self):
        """Post-transform normalization should capture positive variance."""
        X, b = _make_block_tensor()
        M = dct_matrix(X.shape[2])
        result = TBI_I(X, b, M, energy=0.9, max_iter=5)

        assert result.n_iter >= 1
        assert result.total_variance > 0
        assert result.variance_explained[0] > 0


class TestNormOrderingPreTransform:
    """Pre-transform normalization: normalize -> mode3 -> decompose."""

    def test_pre_transform_results_finite(self):
        """Pre-transform normalization should also produce finite, non-NaN results."""
        X, b = _make_block_tensor()
        M = dct_matrix(X.shape[2])

        X_pre = default_normalize(X.copy(), b)
        result = TBI_I(X_pre, b, M, energy=0.9, max_iter=5,
                       normalize_fn=_no_normalize)

        assert np.all(np.isfinite(result.global_scores))
        assert np.all(np.isfinite(result.global_loadings))
        assert np.all(np.isfinite(result.variance_explained))
        assert not np.any(np.isnan(result.variance_explained))

    def test_pre_transform_captures_positive_variance(self):
        """Pre-transform normalization should also capture positive variance."""
        X, b = _make_block_tensor()
        M = dct_matrix(X.shape[2])

        X_pre = default_normalize(X.copy(), b)
        result = TBI_I(X_pre, b, M, energy=0.9, max_iter=5,
                       normalize_fn=_no_normalize)

        assert result.n_iter >= 1
        assert result.total_variance > 0
        assert result.variance_explained[0] > 0


class TestNormOrderingComparison:
    """Compare the two orderings produce structurally consistent results."""

    def test_both_orderings_produce_same_shape_results(self):
        """Both orderings should produce results with matching shapes."""
        X, b = _make_block_tensor()
        M = dct_matrix(X.shape[2])
        max_iter = 3

        result_post = TBI_I(X, b, M, energy=0.99, max_iter=max_iter)

        X_pre = default_normalize(X.copy(), b)
        result_pre = TBI_I(X_pre, b, M, energy=0.99, max_iter=max_iter,
                           normalize_fn=_no_normalize)

        # Both should have run at least 1 iteration
        assert result_post.n_iter >= 1
        assert result_pre.n_iter >= 1

        # Scores and loadings should have the correct leading dimensions
        m, p, n = X.shape
        assert result_post.global_scores.shape[0] == m
        assert result_pre.global_scores.shape[0] == m
        assert result_post.global_loadings.shape[0] == p
        assert result_pre.global_loadings.shape[0] == p

"""
Tests for TBI.metrics — reconstruction error, subspace angles, RV coefficient.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from TBI.metrics import (
    reconstruction_error_from_variance,
    reconstruction_error_tbi_i,
    per_block_variance_contribution,
    subspace_angle,
    rv_coefficient,
)
from TBI import TBI_I, TBI_II
from TBI.analysis_utils import dct_matrix


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_tensor():
    """Small (6, 8, 3) tensor with 2 blocks for integration tests."""
    rng = np.random.RandomState(42)
    m, p, n = 6, 8, 3
    X = rng.randn(m, p, n)
    b = np.array([0, 4])
    M = dct_matrix(n)
    return X, b, M


# ---------------------------------------------------------------------------
# reconstruction_error_from_variance
# ---------------------------------------------------------------------------

class TestReconstructionErrorFromVariance:
    def test_75_percent_captured(self):
        """If 75% of variance is captured, error = sqrt(0.25) = 0.5."""
        var_expl = np.array([60.0, 15.0])  # sum = 75
        total = 100.0
        err = reconstruction_error_from_variance(var_expl, total)
        assert_allclose(err, 0.5, atol=1e-12)

    def test_100_percent_captured(self):
        """Perfect reconstruction -> error = 0."""
        var_expl = np.array([80.0, 20.0])
        total = 100.0
        err = reconstruction_error_from_variance(var_expl, total)
        assert_allclose(err, 0.0, atol=1e-12)

    def test_0_percent_captured(self):
        """No variance captured -> error = 1."""
        var_expl = np.array([0.0])
        total = 100.0
        err = reconstruction_error_from_variance(var_expl, total)
        assert_allclose(err, 1.0, atol=1e-12)

    def test_single_component(self):
        """Single component capturing 50% -> error = sqrt(0.5)."""
        var_expl = np.array([50.0])
        total = 100.0
        err = reconstruction_error_from_variance(var_expl, total)
        assert_allclose(err, np.sqrt(0.5), atol=1e-12)

    def test_degenerate_zero_total(self):
        """Zero total variance -> error = 1.0 (degenerate)."""
        err = reconstruction_error_from_variance(np.array([0.0]), 0.0)
        assert err == 1.0

    def test_negative_total(self):
        """Negative total variance -> error = 1.0 (degenerate)."""
        err = reconstruction_error_from_variance(np.array([1.0]), -5.0)
        assert err == 1.0

    def test_output_in_unit_interval(self):
        """Output is always in [0, 1]."""
        rng = np.random.RandomState(99)
        for _ in range(20):
            total = rng.uniform(1, 1000)
            n_comp = rng.randint(1, 10)
            var_expl = rng.uniform(0, total / n_comp, size=n_comp)
            err = reconstruction_error_from_variance(var_expl, total)
            assert 0.0 <= err <= 1.0


# ---------------------------------------------------------------------------
# reconstruction_error_tbi_i (integration with TBI_I)
# ---------------------------------------------------------------------------

class TestReconstructionErrorTBI_I:
    def test_consistency_with_variance_formula(self, small_tensor):
        """TBI-I reconstruction error matches the variance-based formula."""
        X, b, M = small_tensor
        result = TBI_I(X, b, M, energy=0.99, max_iter=5)

        err = reconstruction_error_tbi_i(X, result, b, M)
        expected = reconstruction_error_from_variance(
            result.variance_explained, result.total_variance
        )
        assert_allclose(err, expected, atol=1e-14)

    def test_error_decreases_with_components(self, small_tensor):
        """More components should yield lower reconstruction error."""
        X, b, M = small_tensor
        errors = []
        for max_iter in [1, 3, 5]:
            result = TBI_I(X, b, M, energy=1.0, max_iter=max_iter)
            errors.append(reconstruction_error_tbi_i(X, result, b, M))

        # Each successive run should have error <= previous
        for i in range(len(errors) - 1):
            assert errors[i] >= errors[i + 1] - 1e-10


# ---------------------------------------------------------------------------
# per_block_variance_contribution
# ---------------------------------------------------------------------------

class TestPerBlockVarianceContribution:
    def test_sums_to_one(self, small_tensor):
        """Block variance fractions sum to approximately 1."""
        X, b, M = small_tensor
        result = TBI_I(X, b, M, energy=0.99, max_iter=5)
        fracs = per_block_variance_contribution(X, result, b, M)

        assert fracs.shape == (len(b),)
        assert_allclose(fracs.sum(), 1.0, atol=1e-10)

    def test_all_nonnegative(self, small_tensor):
        """All block fractions are non-negative."""
        X, b, M = small_tensor
        result = TBI_I(X, b, M, energy=0.99, max_iter=5)
        fracs = per_block_variance_contribution(X, result, b, M)

        assert np.all(fracs >= 0)

    def test_tbi_ii(self, small_tensor):
        """Also works with TBI-II results."""
        X, b, M = small_tensor
        result = TBI_II(X, b, M, energy=0.99, max_iter=5)
        fracs = per_block_variance_contribution(X, result, b, M)

        assert fracs.shape == (len(b),)
        assert_allclose(fracs.sum(), 1.0, atol=1e-10)

    def test_single_block_is_one(self):
        """With a single block, that block captures 100% of variance."""
        rng = np.random.RandomState(7)
        X = rng.randn(5, 6, 3)
        b = np.array([0])
        M = dct_matrix(3)
        result = TBI_I(X, b, M, energy=0.99, max_iter=3)
        fracs = per_block_variance_contribution(X, result, b, M)

        assert fracs.shape == (1,)
        assert_allclose(fracs[0], 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# subspace_angle
# ---------------------------------------------------------------------------

class TestSubspaceAngle:
    def test_identical_subspaces(self):
        """Angle between identical subspaces is 0."""
        rng = np.random.RandomState(123)
        U = rng.randn(10, 3)
        angles = subspace_angle(U, U)

        assert angles.shape == (3,)
        assert_allclose(angles, 0.0, atol=1e-7)

    def test_orthogonal_subspaces(self):
        """Angle between orthogonal subspaces is pi/2."""
        # Use columns of identity to guarantee orthogonality
        U1 = np.eye(6)[:, :2]  # span of e1, e2
        U2 = np.eye(6)[:, 2:4]  # span of e3, e4

        angles = subspace_angle(U1, U2)
        assert angles.shape == (2,)
        assert_allclose(angles, np.pi / 2, atol=1e-10)

    def test_known_angle(self):
        """Rotate a 1D subspace by 45 degrees."""
        U1 = np.array([[1.0], [0.0]])
        U2 = np.array([[np.cos(np.pi / 4)], [np.sin(np.pi / 4)]])

        angles = subspace_angle(U1, U2)
        assert angles.shape == (1,)
        assert_allclose(angles[0], np.pi / 4, atol=1e-10)

    def test_scaled_columns_same_subspace(self):
        """Scaling columns does not change the subspace."""
        rng = np.random.RandomState(77)
        U = rng.randn(8, 2)
        U_scaled = U * np.array([3.0, 0.5])

        angles = subspace_angle(U, U_scaled)
        assert_allclose(angles, 0.0, atol=1e-10)

    def test_different_dimensions(self):
        """Works when U1 and U2 have different numbers of columns."""
        U1 = np.eye(5)[:, :3]
        U2 = np.eye(5)[:, :2]

        angles = subspace_angle(U1, U2)
        # min(3, 2) = 2 angles
        assert angles.shape == (2,)
        # U2 is a subspace of U1, so angles should be 0
        assert_allclose(angles, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# rv_coefficient
# ---------------------------------------------------------------------------

class TestRVCoefficient:
    def test_identical_matrices(self):
        """RV coefficient between identical matrices is 1."""
        rng = np.random.RandomState(10)
        X = rng.randn(8, 3)
        rv = rv_coefficient(X, X)
        assert_allclose(rv, 1.0, atol=1e-10)

    def test_scaled_matrices(self):
        """Scaling does not change RV coefficient (it is scale-invariant)."""
        rng = np.random.RandomState(11)
        X = rng.randn(8, 3)
        rv = rv_coefficient(X, 5.0 * X)
        assert_allclose(rv, 1.0, atol=1e-10)

    def test_orthogonal_configurations(self):
        """RV between orthogonal score sets should be low."""
        U1 = np.eye(6)[:, :2]
        U2 = np.eye(6)[:, 2:4]
        rv = rv_coefficient(U1, U2)
        assert_allclose(rv, 0.0, atol=1e-10)

    def test_in_unit_interval(self):
        """RV coefficient is always in [0, 1]."""
        rng = np.random.RandomState(12)
        for _ in range(20):
            X = rng.randn(10, 4)
            Y = rng.randn(10, 3)
            rv = rv_coefficient(X, Y)
            assert 0.0 - 1e-10 <= rv <= 1.0 + 1e-10

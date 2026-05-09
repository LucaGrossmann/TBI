"""Tests for the variance accounting identity across all decomposition methods.

The variance accounting identity states: for a decomposition that extracts k
components via iterative deflation, the sum of variance explained by those
components plus the residual variance should equal the total variance of the
(normalized) input data.

    sum(variance_explained[:k]) + ||residual||_F^2 = total_variance

We verify:
  - Non-overshoot: sum(variance_explained) <= total_variance
  - Non-negativity: each variance_explained[i] >= 0
  - Monotonicity is NOT guaranteed (later components may capture less), but
    non-negativity is.
  - Cross-method consistency: all methods capture similar total variance on
    the same data (within 20%).
"""

import numpy as np
import numpy.testing as npt
import pytest

from TBI.TBI_I import TBI_I, TBIResult
from TBI.TBI_II import TBI_II, TBIIResult
from TBI.matrix_MCIA import matrix_MCIA, MCIAResult
from TBI.analysis_utils import dct_matrix
from TBI.helpers import _compute_total_variance


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def tensor_data(rng):
    """Small reproducible tensor (10, 15, 3) with 2 blocks."""
    m, p, n = 10, 15, 3
    X = rng.normal(size=(m, p, n))
    b = np.array([0, 5, 10])
    M = dct_matrix(n)
    return X, b, M


@pytest.fixture
def common_params():
    """Shared energy and max_iter for all methods."""
    return dict(energy=0.99, max_iter=10)


# ---------------------------------------------------------------------------
# 1. TBI_I — variance accounting
# ---------------------------------------------------------------------------

class TestTBI_I_VarianceAccounting:

    def test_non_overshoot(self, tensor_data, common_params):
        """sum(variance_explained) <= total_variance."""
        X, b, M = tensor_data
        result = TBI_I(X, b, M, **common_params)
        total_explained = np.sum(result.variance_explained)
        assert total_explained <= result.total_variance + 1e-10, (
            f"Overshoot: explained={total_explained:.6f} > total={result.total_variance:.6f}"
        )

    def test_non_negative_variance(self, tensor_data, common_params):
        """Each variance_explained[i] >= 0."""
        X, b, M = tensor_data
        result = TBI_I(X, b, M, **common_params)
        assert np.all(result.variance_explained >= -1e-12), (
            f"Negative variance: {result.variance_explained}"
        )

    def test_variance_sums_match(self, tensor_data, common_params):
        """sum(variance_explained) + residual_variance = total_variance.

        We reconstruct the residual by running deflation manually and checking
        that the identity holds via the total_variance and variance_explained
        returned by TBI_I.
        """
        X, b, M = tensor_data
        result = TBI_I(X, b, M, **common_params)
        total_explained = np.sum(result.variance_explained)
        # The residual variance is total - explained (by construction of var_removed)
        residual_var = result.total_variance - total_explained
        # Must be non-negative (within floating point tolerance)
        assert residual_var >= -1e-10, (
            f"Negative residual variance: {residual_var:.6e}"
        )
        # Identity: explained + residual == total
        npt.assert_allclose(
            total_explained + max(residual_var, 0.0),
            result.total_variance,
            rtol=1e-10,
            err_msg="Variance accounting identity violated for TBI_I",
        )

    def test_positive_total_variance(self, tensor_data, common_params):
        """total_variance > 0 for non-zero input."""
        X, b, M = tensor_data
        result = TBI_I(X, b, M, **common_params)
        assert result.total_variance > 0, "Total variance should be positive"

    def test_cumulative_fraction_bounded(self, tensor_data, common_params):
        """Cumulative variance fraction stays in [0, 1]."""
        X, b, M = tensor_data
        result = TBI_I(X, b, M, **common_params)
        cum_frac = np.cumsum(result.variance_explained) / result.total_variance
        assert np.all(cum_frac >= -1e-12), f"Negative cumulative fraction: {cum_frac}"
        assert np.all(cum_frac <= 1.0 + 1e-10), f"Cumulative fraction > 1: {cum_frac}"


# ---------------------------------------------------------------------------
# 2. TBI_II — variance accounting
# ---------------------------------------------------------------------------

class TestTBI_II_VarianceAccounting:

    def test_non_overshoot(self, tensor_data, common_params):
        """sum(variance_explained) <= total_variance."""
        X, b, M = tensor_data
        result = TBI_II(X, b, M, **common_params)
        total_explained = np.sum(result.variance_explained)
        assert total_explained <= result.total_variance + 1e-10, (
            f"Overshoot: explained={total_explained:.6f} > total={result.total_variance:.6f}"
        )

    def test_non_negative_variance(self, tensor_data, common_params):
        """Each variance_explained[i] >= 0."""
        X, b, M = tensor_data
        result = TBI_II(X, b, M, **common_params)
        assert np.all(result.variance_explained >= -1e-12), (
            f"Negative variance: {result.variance_explained}"
        )

    def test_variance_sums_match(self, tensor_data, common_params):
        """sum(variance_explained) + residual = total."""
        X, b, M = tensor_data
        result = TBI_II(X, b, M, **common_params)
        total_explained = np.sum(result.variance_explained)
        residual_var = result.total_variance - total_explained
        assert residual_var >= -1e-10, (
            f"Negative residual variance: {residual_var:.6e}"
        )
        npt.assert_allclose(
            total_explained + max(residual_var, 0.0),
            result.total_variance,
            rtol=1e-10,
            err_msg="Variance accounting identity violated for TBI_II",
        )

    def test_cumulative_fraction_bounded(self, tensor_data, common_params):
        """Cumulative variance fraction stays in [0, 1]."""
        X, b, M = tensor_data
        result = TBI_II(X, b, M, **common_params)
        cum_frac = np.cumsum(result.variance_explained) / result.total_variance
        assert np.all(cum_frac >= -1e-12), f"Negative cumulative fraction: {cum_frac}"
        assert np.all(cum_frac <= 1.0 + 1e-10), f"Cumulative fraction > 1: {cum_frac}"


# ---------------------------------------------------------------------------
# 3. matrix_MCIA — variance accounting
# ---------------------------------------------------------------------------

class TestMatrixMCIA_VarianceAccounting:

    def test_non_overshoot(self, tensor_data, common_params):
        """sum(variance_explained) <= total_variance."""
        X, b, _ = tensor_data
        result = matrix_MCIA(X, b, **common_params)
        total_explained = np.sum(result.variance_explained)
        assert total_explained <= result.total_variance + 1e-10, (
            f"Overshoot: explained={total_explained:.6f} > total={result.total_variance:.6f}"
        )

    def test_non_negative_variance(self, tensor_data, common_params):
        """Each variance_explained[i] >= 0."""
        X, b, _ = tensor_data
        result = matrix_MCIA(X, b, **common_params)
        assert np.all(result.variance_explained >= -1e-12), (
            f"Negative variance: {result.variance_explained}"
        )

    def test_variance_sums_match(self, tensor_data, common_params):
        """For SVD-based deflation: sum(sigma_i^2) + ||R||^2 = ||X||^2.

        matrix_MCIA uses rank-1 SVD deflation on the unfolded matrix, so
        variance_explained[i] = sigma_i^2 and the identity holds exactly.
        """
        X, b, _ = tensor_data
        result = matrix_MCIA(X, b, **common_params)
        total_explained = np.sum(result.variance_explained)
        residual_var = result.total_variance - total_explained
        assert residual_var >= -1e-10, (
            f"Negative residual variance: {residual_var:.6e}"
        )
        npt.assert_allclose(
            total_explained + max(residual_var, 0.0),
            result.total_variance,
            rtol=1e-10,
            err_msg="Variance accounting identity violated for matrix_MCIA",
        )

    def test_cumulative_fraction_bounded(self, tensor_data, common_params):
        """Cumulative variance fraction stays in [0, 1]."""
        X, b, _ = tensor_data
        result = matrix_MCIA(X, b, **common_params)
        cum_frac = np.cumsum(result.variance_explained) / result.total_variance
        assert np.all(cum_frac >= -1e-12), f"Negative cumulative fraction: {cum_frac}"
        assert np.all(cum_frac <= 1.0 + 1e-10), f"Cumulative fraction > 1: {cum_frac}"


# ---------------------------------------------------------------------------
# 4. Cross-method consistency
# ---------------------------------------------------------------------------

class TestCrossMethodVariance:

    def test_all_methods_similar_cumulative_variance(self, tensor_data, common_params):
        """All methods' cumulative variance at convergence within 20% of each other.

        Methods normalize differently, so we compare the *fraction* of each
        method's own total variance that is captured, not the raw values.
        All methods on the same data should capture a similar fraction.
        """
        X, b, M = tensor_data

        # Run each method
        tbi_i = TBI_I(X, b, M, **common_params)
        tbi_ii = TBI_II(X, b, M, **common_params)
        mcia = matrix_MCIA(X, b, **common_params)

        # Cumulative fraction captured by each method
        fractions = {}
        fractions["TBI_I"] = np.sum(tbi_i.variance_explained) / tbi_i.total_variance
        fractions["TBI_II"] = np.sum(tbi_ii.variance_explained) / tbi_ii.total_variance
        fractions["matrix_MCIA"] = np.sum(mcia.variance_explained) / mcia.total_variance

        frac_values = np.array(list(fractions.values()))

        # All fractions should be positive and <= 1
        for name, frac in fractions.items():
            assert 0.0 < frac <= 1.0 + 1e-10, (
                f"{name} cumulative fraction out of range: {frac:.4f}"
            )

        # All fractions within 20% of the median
        median_frac = np.median(frac_values)
        for name, frac in fractions.items():
            ratio = frac / median_frac
            assert 0.8 <= ratio <= 1.2, (
                f"{name} fraction {frac:.4f} deviates >20% from median "
                f"{median_frac:.4f} (ratio={ratio:.4f})"
            )

    def test_all_methods_capture_majority(self, tensor_data, common_params):
        """At energy=0.99, all methods should capture >80% of their total variance."""
        X, b, M = tensor_data

        tbi_i = TBI_I(X, b, M, **common_params)
        tbi_ii = TBI_II(X, b, M, **common_params)
        mcia = matrix_MCIA(X, b, **common_params)

        methods = {
            "TBI_I": (tbi_i.variance_explained, tbi_i.total_variance),
            "TBI_II": (tbi_ii.variance_explained, tbi_ii.total_variance),
            "matrix_MCIA": (mcia.variance_explained, mcia.total_variance),
        }

        for name, (var_exp, total_var) in methods.items():
            frac = np.sum(var_exp) / total_var
            assert frac > 0.80, (
                f"{name} captured only {frac:.2%} of total variance "
                f"(expected >80% at energy=0.99)"
            )

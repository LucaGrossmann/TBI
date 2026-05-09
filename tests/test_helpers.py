"""Tests for TBI.helpers — input validation, block utilities, tensor sums, etc."""

import numpy as np
import numpy.testing as npt
import pytest

from TBI.helpers import (
    check_input,
    check_block,
    _validate_inputs,
    _block_ranges,
    _compute_total_variance,
    tensor_sum,
    randorth,
    Mtran,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def valid_inputs(rng):
    """Return a valid (X, b, M) triple for check_input."""
    m, p, n = 8, 6, 3
    X = rng.normal(size=(m, p, n))
    b = np.array([0, 2, 4])
    M = np.eye(n)
    return X, b, M


# ===========================================================================
# check_input
# ===========================================================================

class TestCheckInput:
    """Tests for check_input(X, b, M)."""

    def test_valid_inputs_pass(self, valid_inputs):
        """Valid inputs should not raise."""
        X, b, M = valid_inputs
        check_input(X, b, M)  # should not raise

    def test_non_3d_X(self, valid_inputs):
        """X must be a 3-way tensor."""
        _, b, M = valid_inputs
        X_2d = np.zeros((8, 6))
        with pytest.raises(ValueError, match="not a 3 way tensor"):
            check_input(X_2d, b, M)

    def test_4d_X(self, valid_inputs):
        _, b, M = valid_inputs
        X_4d = np.zeros((2, 3, 3, 4))
        with pytest.raises(ValueError, match="not a 3 way tensor"):
            check_input(X_4d, b, M)

    def test_non_1d_b(self, valid_inputs):
        X, _, M = valid_inputs
        b_2d = np.array([[0, 2]])
        with pytest.raises(ValueError, match="not a vector"):
            check_input(X, b_2d, M)

    def test_non_2d_M(self, valid_inputs):
        X, b, _ = valid_inputs
        M_1d = np.array([1.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="not a matrix"):
            check_input(X, b, M_1d)

    def test_dimension_mismatch_X_M(self, valid_inputs):
        X, b, _ = valid_inputs
        M_wrong = np.eye(5)  # n=3, but M is 5x5
        with pytest.raises(ValueError, match="incompatible dimensions"):
            check_input(X, b, M_wrong)

    def test_non_square_M(self, valid_inputs):
        X, b, _ = valid_inputs
        M_rect = np.ones((3, 4))
        with pytest.raises(ValueError, match="not a square matrix"):
            check_input(X, b, M_rect)

    def test_negative_b(self, valid_inputs):
        X, _, M = valid_inputs
        b_neg = np.array([-1, 2, 4])
        with pytest.raises(ValueError, match="negative"):
            check_input(X, b_neg, M)

    def test_empty_b(self, valid_inputs):
        X, _, M = valid_inputs
        b_empty = np.array([], dtype=int)
        with pytest.raises(ValueError, match="at least one"):
            check_input(X, b_empty, M)


# ===========================================================================
# check_block
# ===========================================================================

class TestCheckBlock:
    """Tests for check_block(b, p)."""

    def test_valid_single_block(self):
        check_block(np.array([0]), p=10)  # no error

    def test_valid_multiple_blocks(self):
        check_block(np.array([0, 3, 7]), p=10)  # no error

    def test_empty_b_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            check_block(np.array([], dtype=int), p=10)

    def test_non_1d_raises(self):
        with pytest.raises(ValueError, match="1D"):
            check_block(np.array([[0, 3]]), p=10)

    def test_non_integer_raises(self):
        with pytest.raises(TypeError, match="integers"):
            check_block(np.array([0.0, 3.0]), p=10)

    def test_negative_values_raises(self):
        with pytest.raises(ValueError, match=">= 0"):
            check_block(np.array([-1, 3]), p=10)

    def test_b_exceeds_p_raises(self):
        with pytest.raises(ValueError, match="< p"):
            check_block(np.array([0, 10]), p=10)

    def test_unsorted_raises(self):
        with pytest.raises(ValueError, match="sorted"):
            check_block(np.array([0, 5, 3]), p=10)

    def test_duplicate_raises(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            check_block(np.array([0, 3, 3, 7]), p=10)

    def test_not_starting_at_zero_raises(self):
        with pytest.raises(ValueError, match="b\\[0\\].*must be 0"):
            check_block(np.array([1, 3, 7]), p=10)


# ===========================================================================
# _validate_inputs
# ===========================================================================

class TestValidateInputs:
    """_validate_inputs delegates to check_input + check_block."""

    def test_valid_passes(self, valid_inputs):
        X, b, M = valid_inputs
        _validate_inputs(X, b, M)  # no error

    def test_invalid_X_shape_raises(self, valid_inputs):
        _, b, M = valid_inputs
        X_bad = np.zeros((5, 6))
        with pytest.raises(ValueError):
            _validate_inputs(X_bad, b, M)

    def test_invalid_block_raises(self, valid_inputs):
        X, _, M = valid_inputs
        b_bad = np.array([1, 3])  # doesn't start at 0
        with pytest.raises(ValueError):
            _validate_inputs(X, b_bad, M)


# ===========================================================================
# _block_ranges
# ===========================================================================

class TestBlockRanges:
    """_block_ranges yields (idx, start, end) tuples."""

    def test_single_block(self):
        b = np.array([0])
        ranges = list(_block_ranges(b, p=10))
        assert ranges == [(0, 0, 10)]

    def test_two_blocks(self):
        b = np.array([0, 5])
        ranges = list(_block_ranges(b, p=10))
        assert ranges == [(0, 0, 5), (1, 5, 10)]

    def test_three_blocks(self):
        b = np.array([0, 3, 7])
        ranges = list(_block_ranges(b, p=12))
        assert ranges == [(0, 0, 3), (1, 3, 7), (2, 7, 12)]

    def test_contiguous_coverage(self):
        """Blocks should cover [0, p) without gaps or overlaps."""
        b = np.array([0, 4, 9, 15])
        p = 20
        ranges = list(_block_ranges(b, p))
        # Full coverage check
        covered = set()
        for _, start, end in ranges:
            covered.update(range(start, end))
        assert covered == set(range(p))


# ===========================================================================
# _compute_total_variance
# ===========================================================================

class TestComputeTotalVariance:
    """_compute_total_variance returns sum of squared elements."""

    def test_known_value(self):
        X = np.array([[[1.0, 2.0], [3.0, 4.0]]])
        # 1 + 4 + 9 + 16 = 30
        assert _compute_total_variance(X) == pytest.approx(30.0)

    def test_zero_tensor(self):
        X = np.zeros((3, 4, 2))
        assert _compute_total_variance(X) == pytest.approx(0.0)

    def test_matches_frob_norm_squared(self, rng):
        X = rng.normal(size=(5, 6, 3))
        expected = np.sum(X ** 2)
        assert _compute_total_variance(X) == pytest.approx(expected)


# ===========================================================================
# tensor_sum
# ===========================================================================

class TestTensorSum:
    """tensor_sum returns (cs, rs, ts) with correct shapes and values."""

    def test_shapes(self):
        m, p_k, n = 4, 5, 3
        X_k = np.ones((m, p_k, n))
        cs, rs, ts = tensor_sum(X_k)
        assert cs.shape == (1, p_k, 1)
        assert rs.shape == (m, p_k, n)
        assert isinstance(float(ts), float)

    def test_known_values(self):
        # 2x3x2 tensor of ones: each element = 1
        X_k = np.ones((2, 3, 2))
        cs, rs, ts = tensor_sum(X_k)

        # Column sum: sum over samples (2) and sheets (2) -> 4 per variable
        npt.assert_array_equal(cs, np.full((1, 3, 1), 4.0))

        # Row sum: sum over variables (3) per (sample, sheet) -> 3, repeated 3 times
        npt.assert_array_equal(rs, np.full((2, 3, 2), 3.0))

        # Total sum: 2 * 3 * 2 = 12
        assert ts == pytest.approx(12.0)

    def test_column_sum_sums_correct_axes(self, rng):
        """Column sums should equal manual sum over axes 0 and 2."""
        X_k = rng.normal(size=(4, 5, 3))
        cs, _, _ = tensor_sum(X_k)
        expected = np.sum(X_k, axis=(0, 2)).reshape(1, 5, 1)
        npt.assert_allclose(cs, expected)

    def test_row_sum_sums_correct_axis(self, rng):
        """Row sums should equal manual sum over axis 1, broadcast."""
        X_k = rng.normal(size=(4, 5, 3))
        _, rs, _ = tensor_sum(X_k)
        manual_rs = np.sum(X_k, axis=1)  # (4, 3)
        expected = np.repeat(manual_rs[:, None, :], 5, axis=1)
        npt.assert_allclose(rs, expected)


# ===========================================================================
# randorth
# ===========================================================================

class TestRandorth:
    """randorth returns an orthonormal matrix."""

    @pytest.mark.parametrize("n", [1, 2, 3, 5, 10])
    def test_orthogonality(self, n):
        Q = randorth(n)
        npt.assert_allclose(Q.T @ Q, np.eye(n), atol=1e-12)

    @pytest.mark.parametrize("n", [1, 3, 7])
    def test_shape(self, n):
        Q = randorth(n)
        assert Q.shape == (n, n)

    def test_determinant_abs_one(self):
        """|det(Q)| should be 1 for orthogonal Q."""
        Q = randorth(5)
        assert abs(np.linalg.det(Q)) == pytest.approx(1.0, abs=1e-12)


# ===========================================================================
# Mtran
# ===========================================================================

class TestMtran:
    """Mtran transposes + conjugates each frontal slice."""

    def test_shape(self, rng):
        A = rng.normal(size=(4, 6, 3))
        B = Mtran(A)
        assert B.shape == (6, 4, 3)

    def test_real_transpose(self, rng):
        """For real tensors, Mtran == transpose of each slice."""
        A = rng.normal(size=(4, 6, 3))
        B = Mtran(A)
        for s in range(3):
            npt.assert_allclose(B[:, :, s], A[:, :, s].T)

    def test_complex_conjugate_transpose(self):
        """For complex tensors, each slice should be conjugate-transposed."""
        A = np.array([[[1 + 1j, 2 + 2j],
                       [3 + 3j, 4 + 4j]]])  # (1, 2, 2)
        B = Mtran(A)
        assert B.shape == (2, 1, 2)
        for s in range(2):
            npt.assert_allclose(B[:, :, s], A[:, :, s].conj().T)

    def test_double_mtran_identity(self, rng):
        """Mtran(Mtran(A)) should give back A for real-valued tensors."""
        A = rng.normal(size=(5, 5, 3))
        npt.assert_allclose(Mtran(Mtran(A)), A, atol=1e-14)

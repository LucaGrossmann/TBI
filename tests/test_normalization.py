import numpy as np
import numpy.testing as npt
import pytest
from TBI.normalization import (
    variable_mean_center, variable_var_normalize,
    block_var_normalize, block_frob_normalize,
    MCIA_tensor_norm,
)

def test_variable_mean_center_sheet_level_true_zeroes_sheet_means():
    rng = np.random.default_rng(0)
    X = rng.random((4, 4, 4))

    sheet_offsets = np.array([1.0, 2.0, 3.0, 4.0])
    X = X + sheet_offsets.reshape(1, 1, 4)

    Xc = variable_mean_center(X, sheet_level=True)

    # Per-(variable,timepoint) mean across samples should be ~0
    npt.assert_allclose(Xc.mean(axis=0), 0.0, atol=1e-12)

def test_variable_mean_center_sheet_level_false_zeroes_global_variable_means():
    rng = np.random.default_rng(0)
    X = rng.random((4, 4, 4))

    sheet_offsets = np.array([1.0, 2.0, 3.0, 4.0])
    X = X + sheet_offsets.reshape(1, 1, 4)

    Xc = variable_mean_center(X, sheet_level=False)

    # Per-variable mean across samples and timepoints should be ~0
    npt.assert_allclose(Xc.mean(axis=(0, 2)), 0.0, atol=1e-12)

def test_variable_mean_center_preserves_shape_and_dtype():
    rng = np.random.default_rng(1)
    X = rng.random((3, 5, 7)).astype(np.float64)

    Xc1 = variable_mean_center(X, sheet_level=True)
    Xc2 = variable_mean_center(X, sheet_level=False)

    assert Xc1.shape == X.shape
    assert Xc2.shape == X.shape
    assert Xc1.dtype == X.dtype
    assert Xc2.dtype == X.dtype


def test_variable_mean_center_is_idempotent():
    rng = np.random.default_rng(2)
    X = rng.random((4, 4, 4))

    Xc = variable_mean_center(X, sheet_level=False)
    Xcc = variable_mean_center(Xc, sheet_level=False)

    npt.assert_allclose(Xcc, Xc, atol=1e-12)


def test_var_normalize_global_shape_preserved():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(7, 5, 11))
    Y = variable_var_normalize(X, sheet_level=False)
    assert Y.shape == X.shape


def test_var_normalize_sheet_level_shape_preserved():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(10, 3, 4))
    Y = variable_var_normalize(X, sheet_level=True)
    assert Y.shape == X.shape


def test_var_normalize_global_makes_unit_std_per_variable():
    rng = np.random.default_rng(2)
    m, p, n = 200, 4, 30

    # Give each variable a different std (global across samples/timepoints)
    scales = np.array([0.5, 2.0, 3.0, 1.5]).reshape(1, p, 1)
    X = rng.normal(size=(m, p, n)) * scales

    Y = variable_var_normalize(X, sheet_level=False)

    # std across (samples,timepoints) for each variable should be ~1
    std_after = np.std(Y, axis=(0, 2), ddof=0)  # shape (p,)
    assert np.allclose(std_after, np.ones(p), atol=1e-2)


def test_var_normalize_sheet_level_makes_unit_std_per_variable_timepoint():
    rng = np.random.default_rng(3)
    m, p, n = 300, 3, 5

    # Different std per (variable,timepoint)
    scales = rng.uniform(0.3, 3.0, size=(p, n)).reshape(1, p, n)
    X = rng.normal(size=(m, p, n)) * scales

    Y = variable_var_normalize(X, sheet_level=True)

    std_after = np.std(Y, axis=0, ddof=0)  # shape (p,n)
    assert np.allclose(std_after, np.ones((p, n)), atol=2e-2)


def test_var_normalize_handles_zero_variance_no_nan_inf():
    X = np.ones((8, 3, 6)) * 7.0  # constant -> std = 0 everywhere
    Y = variable_var_normalize(X, sheet_level=False)

    assert np.isfinite(Y).all()
    # since we treat std=0 as std=1, output should equal input
    assert np.allclose(Y, X)



def _block_slices(b, p):
    """Yield (start, end) pairs for blocks defined by b."""
    b = np.asarray(b, dtype=int)
    for i, start in enumerate(b):
        end = b[i + 1] if (i + 1 < b.size) else p
        yield start, end


@pytest.mark.parametrize("shape,b", [
    ((7, 8, 5), np.array([0, 2, 5])),  # 3 blocks
    ((3, 10, 4), np.array([0, 3, 9])), # uneven blocks
])
def test_sheet_level_false_block_variance_is_one(shape, b):
    """
    sheet_level=False: each block should have unit variance (ddof=0)
    over all entries in that block (axes 0,1,2 restricted to block).
    """
    rng = np.random.default_rng(0)
    X = rng.normal(size=shape) * 3.7 + 1.2  # non-unit scale, nonzero mean

    Y = block_var_normalize(X, b, sheet_level=False, ddof=0, eps=1e-12, copy=True)

    m, p, n = Y.shape
    for start, end in _block_slices(b, p):
        blk = Y[:, start:end, :]  # (m, p_blk, n)
        v = np.var(blk, ddof=0)
        np.testing.assert_allclose(v, 1.0, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("shape,b", [
    ((7, 8, 5), np.array([0, 2, 5])),
    ((4, 6, 9), np.array([0, 1, 4])),
])
def test_sheet_level_true_block_variance_per_sheet_is_one(shape, b):
    """
    sheet_level=True: for each sheet/timepoint j, each block should have
    unit variance across axes (subjects, variables_in_block) == (0,1).
    """
    rng = np.random.default_rng(1)
    X = rng.normal(size=shape) * 5.0 + 2.0

    Y = block_var_normalize(X, b, sheet_level=True, ddof=0, eps=1e-12, copy=True)

    m, p, n = Y.shape
    for start, end in _block_slices(b, p):
        blk = Y[:, start:end, :]  # (m, p_blk, n)
        # variance per sheet: shape (n,)
        v = np.var(blk, axis=(0, 1), ddof=0)
        np.testing.assert_allclose(v, np.ones(n), rtol=1e-6, atol=1e-6)


def test_copy_flag_behavior():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(5, 8, 4))
    b = np.array([0, 3, 6])

    X_orig = X.copy()
    Y = block_var_normalize(X, b, copy=True)
    # copy=True should not modify X
    np.testing.assert_allclose(X, X_orig)
    # and should return a different array object
    assert Y is not X

    # copy=False should modify in-place and return same object
    X2 = X_orig.copy()
    Y2 = block_var_normalize(X2, b, copy=False)
    assert Y2 is X2
    assert not np.allclose(X2, X_orig)


def test_eps_guards_zero_variance_block_leaves_it_unchanged():
    """
    If a block has ~zero std, safe_std should become 1 and block should remain unchanged.
    """
    rng = np.random.default_rng(3)
    m, p, n = 4, 8, 5
    X = rng.normal(size=(m, p, n))

    # Make the last block constant => variance 0
    b = np.array([0, 2, 5])  # blocks: [0:2], [2:5], [5:8]
    X[:, 5:8, :] = 7.0

    Y = block_var_normalize(X, b, sheet_level=False, eps=1e-12, copy=True)

    # constant block should remain constant (unchanged)
    np.testing.assert_allclose(Y[:, 5:8, :], X[:, 5:8, :])


def test_b_can_be_python_list():
    """
    Useful regression test if you coerce b via np.asarray(b, dtype=int).
    If your current implementation doesn't, this test will fail (good signal).
    """
    rng = np.random.default_rng(4)
    X = rng.normal(size=(3, 8, 2))
    b = [0, 4]  # list on purpose

    Y = block_var_normalize(X, b, sheet_level=False, copy=True)
    assert Y.shape == X.shape


def test_invalid_X_dim_raises():
    X = np.zeros((3, 4))  # not 3D
    b = np.array([0, 2])
    with pytest.raises(ValueError):
        block_var_normalize(X, b)


# ---------------------------------------------------------------------------
# block_frob_normalize tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape,b", [
    ((7, 8, 5), np.array([0, 2, 5])),
    ((3, 10, 4), np.array([0, 3, 9])),
    ((5, 6, 3), np.array([0, 2, 4])),
])
def test_frob_normalize_equal_block_variance(shape, b):
    """After Frobenius normalization, every block should have Frobenius norm = 1."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=shape) * rng.uniform(0.5, 10.0, size=(1, shape[1], 1))

    Y = block_frob_normalize(X, b)

    _, p, _ = Y.shape
    for start, end in _block_slices(b, p):
        fnorm = np.sqrt(np.sum(Y[:, start:end, :] ** 2))
        npt.assert_allclose(fnorm, 1.0, rtol=1e-10)


def test_frob_normalize_blocks_contribute_equally():
    """Sum of squares per block should be identical after Frobenius normalization."""
    rng = np.random.default_rng(7)
    m, p, n = 10, 20, 4
    X = rng.normal(size=(m, p, n))
    # Give blocks very different scales
    X[:, :5, :] *= 100.0
    X[:, 5:15, :] *= 0.01
    b = np.array([0, 5, 15])

    Y = block_frob_normalize(X, b)

    var0 = np.sum(Y[:, 0:5, :] ** 2)
    var1 = np.sum(Y[:, 5:15, :] ** 2)
    var2 = np.sum(Y[:, 15:20, :] ** 2)
    npt.assert_allclose(var0, var1, rtol=1e-10)
    npt.assert_allclose(var1, var2, rtol=1e-10)


def test_frob_normalize_preserves_shape():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4, 8, 3))
    b = np.array([0, 3, 6])
    Y = block_frob_normalize(X, b)
    assert Y.shape == X.shape


def test_frob_normalize_copy_flag():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4, 8, 3))
    b = np.array([0, 3])

    X_orig = X.copy()
    Y = block_frob_normalize(X, b, copy=True)
    npt.assert_allclose(X, X_orig)
    assert Y is not X

    X2 = X_orig.copy()
    Y2 = block_frob_normalize(X2, b, copy=False)
    assert Y2 is X2


def test_frob_normalize_zero_block_unchanged():
    """A block of all zeros should be left as zeros."""
    X = np.zeros((3, 6, 2))
    X[:, :3, :] = 1.0  # first block non-zero
    b = np.array([0, 3])

    Y = block_frob_normalize(X, b)
    npt.assert_allclose(Y[:, 3:, :], 0.0)


def test_frob_normalize_invalid_X_raises():
    X = np.zeros((3, 4))
    b = np.array([0, 2])
    with pytest.raises(ValueError):
        block_frob_normalize(X, b)



# ---------------------------------------------------------------------------
# MCIA_tensor_norm tests
# ---------------------------------------------------------------------------

def test_mcia_preserves_shape():
    rng = np.random.default_rng(0)
    X = rng.random((5, 8, 3))
    b = np.array([0, 3, 6])
    Y = MCIA_tensor_norm(X, b)
    assert Y.shape == X.shape


def test_mcia_output_is_approximately_mean_centered():
    """CA standardized residuals should be approximately mean-centered per variable."""
    rng = np.random.default_rng(42)
    m, p, n = 50, 10, 4
    X = rng.random((m, p, n)) * 10 + 5
    b = np.array([0, 4, 7])

    Y = MCIA_tensor_norm(X, b)

    # Weighted mean across samples for each (variable, sheet) should be near zero
    # Use simple mean as a proxy — CA residuals sum to zero by construction
    var_means = Y.mean(axis=(0, 2))  # shape (p,)
    npt.assert_allclose(var_means, 0.0, atol=0.5)


def test_mcia_handles_zero_block():
    """A block of all zeros should produce zeros, not NaN/Inf."""
    X = np.zeros((4, 6, 3))
    X[:, :3, :] = np.random.default_rng(0).random((4, 3, 3))
    b = np.array([0, 3])

    Y = MCIA_tensor_norm(X, b)

    assert np.isfinite(Y).all()
    npt.assert_allclose(Y[:, 3:, :], 0.0)


def test_mcia_handles_negative_input():
    """Negative input should be offset to positive; no NaN/Inf in output."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(6, 8, 3))  # has negative values
    b = np.array([0, 4])

    Y = MCIA_tensor_norm(X, b)
    assert np.isfinite(Y).all()


def test_mcia_no_artificial_dominance():
    """
    First SVD component should NOT capture >95% variance on random data.
    The old formula produced ~99.9% on first component because it wasn't mean-centered.
    """
    rng = np.random.default_rng(99)
    m, p, n = 30, 20, 5
    X = rng.random((m, p, n)) * 10 + 1
    b = np.array([0, 8, 15])

    Y = MCIA_tensor_norm(X, b)

    # Check first sheet's SVD — first singular value should not dominate
    total_var = np.sum(Y ** 2)
    U, s, Vt = np.linalg.svd(Y[:, :, 0], full_matrices=False)
    first_component_var = s[0] ** 2
    ratio = first_component_var / total_var
    assert ratio < 0.95, f"First component captures {ratio:.1%} — still too dominant"


def test_mcia_epsilon_guard():
    """Column sum near zero should not produce NaN/Inf."""
    X = np.ones((4, 6, 2)) * 1e-15  # near-zero values
    X[:, :3, :] = 5.0  # first block has real values
    b = np.array([0, 3])

    Y = MCIA_tensor_norm(X, b)
    assert np.isfinite(Y).all()


def test_mcia_blocks_processed_independently():
    """Each block is normalized using only its own marginals, not the other block's."""
    rng = np.random.default_rng(7)
    m, p, n = 10, 10, 3
    X = rng.random((m, p, n))
    X[:, :5, :] *= 1000.0  # block 0 is much larger
    X[:, 5:, :] *= 0.01    # block 1 is much smaller
    b = np.array([0, 5])

    Y = MCIA_tensor_norm(X, b)

    # Both blocks should produce finite, non-trivial output
    assert np.isfinite(Y).all()
    assert np.var(Y[:, :5, :]) > 1e-15, "Block 0 collapsed to zero"
    assert np.var(Y[:, 5:, :]) > 1e-15, "Block 1 collapsed to zero"

    # Changing block 0 should not affect block 1's output
    X2 = X.copy()
    X2[:, :5, :] *= 3.0  # perturb block 0
    Y2 = MCIA_tensor_norm(X2, b)
    npt.assert_allclose(Y2[:, 5:, :], Y[:, 5:, :],
                        err_msg="Block 1 changed when only block 0 was perturbed")
import numpy as np
import numpy.testing as npt
import pytest
from TBI.star_M import mode3, mode3_fiber, starM, Msvd
from TBI.helpers import Mtran, randorth


# ---------------------------------------------------------------------------
# mode3
# ---------------------------------------------------------------------------

def test_mode3_output_shape():
    rng = np.random.default_rng(0)
    m, p, n, r = 5, 4, 6, 3
    A = rng.random((m, p, n))
    U = rng.random((r, n))
    B = mode3(A, U)
    assert B.shape == (m, p, r)


def test_mode3_identity_is_noop():
    rng = np.random.default_rng(1)
    A = rng.random((4, 3, 5))
    npt.assert_allclose(mode3(A, np.eye(5)), A, atol=1e-12)


def test_mode3_dimension_mismatch_raises():
    A = np.zeros((4, 3, 5))
    U = np.zeros((2, 7))  # second dim doesn't match A.shape[2]
    with pytest.raises(ValueError):
        mode3(A, U)


def test_mode3_known_value():
    # A has shape (1, 1, 2); U doubles the first slice and zeros the second
    A = np.array([[[3.0, 7.0]]])   # (1, 1, 2)
    U = np.array([[2.0, 0.0]])     # (1, 2) -> output is (1, 1, 1)
    B = mode3(A, U)
    assert B.shape == (1, 1, 1)
    npt.assert_allclose(B[0, 0, 0], 6.0, atol=1e-12)


# ---------------------------------------------------------------------------
# mode3_fiber (tube-fiber definition) — must match mode3 exactly
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("m,p,n,r", [
    (5, 4, 6, 3),
    (3, 7, 4, 4),
    (1, 1, 5, 2),
    (6, 3, 8, 8),
])
def test_mode3_fiber_matches_mode3(m, p, n, r):
    rng = np.random.default_rng(42)
    A = rng.random((m, p, n))
    U = rng.random((r, n))
    npt.assert_allclose(mode3_fiber(A, U), mode3(A, U), atol=1e-12)


def test_mode3_fiber_identity_is_noop():
    rng = np.random.default_rng(43)
    A = rng.random((4, 3, 5))
    npt.assert_allclose(mode3_fiber(A, np.eye(5)), A, atol=1e-12)


def test_mode3_fiber_dimension_mismatch_raises():
    A = np.zeros((4, 3, 5))
    U = np.zeros((2, 7))
    with pytest.raises(ValueError):
        mode3_fiber(A, U)


def test_mode3_fiber_matches_mode3_with_orthogonal_M():
    rng = np.random.default_rng(44)
    A = rng.random((5, 4, 6))
    M = randorth(6)
    npt.assert_allclose(mode3_fiber(A, M), mode3(A, M), atol=1e-12)


def test_mode3_fiber_known_value():
    # Same known-value test as mode3: manually verify the fiber computation
    A = np.array([[[3.0, 7.0]]])   # (1, 1, 2)
    U = np.array([[2.0, 0.0]])     # (1, 2) -> U @ [3, 7] = 6
    B = mode3_fiber(A, U)
    assert B.shape == (1, 1, 1)
    npt.assert_allclose(B[0, 0, 0], 6.0, atol=1e-12)


def test_mode3_fiber_matches_mode3_non_square_U():
    # U is rectangular: r != n (more rows than columns, and vice versa)
    rng = np.random.default_rng(45)
    A = rng.random((4, 5, 3))
    U_tall = rng.random((7, 3))    # r > n: expanding
    U_wide = rng.random((2, 3))    # r < n: compressing
    npt.assert_allclose(mode3_fiber(A, U_tall), mode3(A, U_tall), atol=1e-12)
    npt.assert_allclose(mode3_fiber(A, U_wide), mode3(A, U_wide), atol=1e-12)


def test_mode3_fiber_matches_mode3_single_sheet():
    # Edge case: n=1 (single sheet tensor)
    rng = np.random.default_rng(46)
    A = rng.random((3, 4, 1))
    U = rng.random((1, 1))
    npt.assert_allclose(mode3_fiber(A, U), mode3(A, U), atol=1e-12)


def test_mode3_fiber_matches_mode3_large_random():
    # Larger tensor to stress-test equivalence
    rng = np.random.default_rng(47)
    A = rng.random((20, 15, 10))
    U = rng.random((8, 10))
    npt.assert_allclose(mode3_fiber(A, U), mode3(A, U), atol=1e-10)


def test_mode3_fiber_matches_mode3_dct_matrix():
    # Use a DCT-like orthogonal matrix (common in practice)
    from scipy.fft import dct
    n = 6
    M = dct(np.eye(n), type=2, axis=0, norm='ortho')
    rng = np.random.default_rng(48)
    A = rng.random((5, 4, n))
    npt.assert_allclose(mode3_fiber(A, M), mode3(A, M), atol=1e-12)


def test_mode3_fiber_matches_mode3_many_seeds():
    # Run equivalence across 10 different random seeds
    for seed in range(50, 60):
        rng = np.random.default_rng(seed)
        m, p, n, r = rng.integers(2, 10, size=4)
        A = rng.random((m, p, n))
        U = rng.random((r, n))
        npt.assert_allclose(
            mode3_fiber(A, U), mode3(A, U), atol=1e-12,
            err_msg=f"Failed for seed={seed}, shape=({m},{p},{n}), r={r}"
        )


# ---------------------------------------------------------------------------
# starM
# ---------------------------------------------------------------------------

def test_starM_output_shape():
    rng = np.random.default_rng(2)
    m, p, l, n = 5, 4, 3, 6
    A = rng.random((m, p, n))
    B = rng.random((p, l, n))
    M = randorth(n)
    C = starM(A, B, M)
    assert C.shape == (m, l, n)


def test_starM_matches_manual_computation():
    # C should equal: mode3( slicewise(Ahat @ Bhat), M^T )
    rng = np.random.default_rng(3)
    m, p, l, n = 4, 3, 5, 6
    A = rng.random((m, p, n))
    B = rng.random((p, l, n))
    M = randorth(n)

    C = starM(A, B, M)

    Ahat = mode3(A, M)
    Bhat = mode3(B, M)
    Chat = np.zeros((m, l, n))
    for i in range(n):
        Chat[:, :, i] = Ahat[:, :, i] @ Bhat[:, :, i]
    C_ref = mode3(Chat, M.T)

    npt.assert_allclose(C, C_ref, atol=1e-10)


@pytest.mark.parametrize("m,p,l,n", [
    (5, 4, 3, 6),
    (3, 7, 2, 4),
    (1, 1, 1, 1),
    (10, 8, 6, 12),
    (20, 15, 10, 8),
])
def test_starM_vectorized_matches_loop(m, p, l, n):
    rng = np.random.default_rng(60)
    A = rng.random((m, p, n))
    B = rng.random((p, l, n))
    M = randorth(n)
    C_vec  = starM(A, B, M, vectorized=True)
    C_loop = starM(A, B, M, vectorized=False)
    npt.assert_allclose(C_vec, C_loop, atol=1e-10)


def test_starM_vectorized_matches_loop_many_seeds():
    for seed in range(70, 80):
        rng = np.random.default_rng(seed)
        m, p, l, n = rng.integers(2, 8, size=4)
        A = rng.random((m, p, n))
        B = rng.random((p, l, n))
        M = randorth(n)
        C_vec  = starM(A, B, M, vectorized=True)
        C_loop = starM(A, B, M, vectorized=False)
        npt.assert_allclose(
            C_vec, C_loop, atol=1e-10,
            err_msg=f"Failed for seed={seed}, shapes A={A.shape} B={B.shape}"
        )


def test_starM_vectorized_matches_loop_square_tensors():
    rng = np.random.default_rng(80)
    n = 6
    A = rng.random((n, n, n))
    B = rng.random((n, n, n))
    M = randorth(n)
    C_vec  = starM(A, B, M, vectorized=True)
    C_loop = starM(A, B, M, vectorized=False)
    npt.assert_allclose(C_vec, C_loop, atol=1e-10)


def test_starM_non_unitary_M_raises():
    A = np.zeros((3, 3, 4))
    B = np.zeros((3, 3, 4))
    M = np.eye(4) * 2.0  # not unitary (norm ≠ 1)
    with pytest.raises(ValueError, match="not orthogonal"):
        starM(A, B, M)


def test_starM_dimension_mismatch_AB_raises():
    rng = np.random.default_rng(4)
    n = 4
    M = randorth(n)
    A = rng.random((3, 5, n))
    B = rng.random((7, 2, n))  # B.shape[0]=7 != A.shape[1]=5
    with pytest.raises(ValueError):
        starM(A, B, M)


def test_starM_wrong_M_shape_raises():
    rng = np.random.default_rng(5)
    n = 4
    A = rng.random((3, 3, n))
    B = rng.random((3, 3, n))
    M = randorth(n + 1)  # wrong size
    with pytest.raises(ValueError):
        starM(A, B, M)


# ---------------------------------------------------------------------------
# Mtran
# ---------------------------------------------------------------------------

def test_Mtran_output_shape():
    A = np.zeros((5, 3, 7))
    assert Mtran(A).shape == (3, 5, 7)


def test_Mtran_double_transpose_is_identity():
    rng = np.random.default_rng(6)
    A = rng.random((4, 6, 5))
    npt.assert_allclose(Mtran(Mtran(A)), A, atol=1e-12)


def test_Mtran_transposes_each_slice():
    rng = np.random.default_rng(7)
    A = rng.random((4, 6, 5))
    B = Mtran(A)
    for i in range(5):
        npt.assert_allclose(B[:, :, i], A[:, :, i].T, atol=1e-12)


# ---------------------------------------------------------------------------
# Msvd
# ---------------------------------------------------------------------------

def _make_random_tensor(shape, seed):
    rng = np.random.default_rng(seed)
    return rng.random(shape)


@pytest.mark.parametrize("n1,n2,n3", [(5, 4, 6), (4, 5, 3), (3, 3, 4)])
def test_Msvd_full_output_shapes(n1, n2, n3):
    B = _make_random_tensor((n1, n2, n3), seed=10)
    M = randorth(n3)
    U, S, V = Msvd(B, M, compressed=False)
    assert U.shape == (n1, n1, n3)
    assert S.shape == (n1, n2, n3)
    assert V.shape == (n2, n2, n3)


@pytest.mark.parametrize("n1,n2,n3", [(5, 4, 6), (4, 5, 3), (3, 3, 4)])
def test_Msvd_compressed_output_shapes(n1, n2, n3):
    B = _make_random_tensor((n1, n2, n3), seed=11)
    M = randorth(n3)
    k = min(n1, n2)
    U, S, V = Msvd(B, M, compressed=True)
    assert U.shape == (n1, k,  n3)
    assert S.shape == (k,  k,  n3)
    assert V.shape == (k,  n2, n3)


def test_Msvd_full_reconstruction():
    rng = np.random.default_rng(12)
    n1, n2, n3 = 5, 4, 6
    B = rng.random((n1, n2, n3))
    M = randorth(n3)
    B_hat = mode3(B, M)
    U, S, V = Msvd(B, M, compressed=False)
    for i in range(n3):
        reconstructed = U[:, :, i] @ S[:, :, i] @ V[:, :, i]
        npt.assert_allclose(reconstructed, B_hat[:, :, i], atol=1e-10)


def test_Msvd_compressed_reconstruction():
    rng = np.random.default_rng(13)
    n1, n2, n3 = 5, 4, 6
    B = rng.random((n1, n2, n3))
    M = randorth(n3)
    B_hat = mode3(B, M)
    U, S, V = Msvd(B, M, compressed=True)
    for i in range(n3):
        reconstructed = U[:, :, i] @ S[:, :, i] @ V[:, :, i]
        npt.assert_allclose(reconstructed, B_hat[:, :, i], atol=1e-10)


def test_Msvd_full_U_is_orthonormal_per_slice():
    rng = np.random.default_rng(14)
    n1, n2, n3 = 5, 4, 6
    B = rng.random((n1, n2, n3))
    M = randorth(n3)
    U, _, _ = Msvd(B, M, compressed=False)
    for i in range(n3):
        npt.assert_allclose(U[:, :, i].T @ U[:, :, i], np.eye(n1), atol=1e-10)


def test_Msvd_full_V_rows_are_orthonormal_per_slice():
    rng = np.random.default_rng(15)
    n1, n2, n3 = 5, 4, 6
    B = rng.random((n1, n2, n3))
    M = randorth(n3)
    _, _, V = Msvd(B, M, compressed=False)
    for i in range(n3):
        # V stores V^H; rows of V^H are orthonormal => V @ V.T = I
        npt.assert_allclose(V[:, :, i] @ V[:, :, i].T, np.eye(n2), atol=1e-10)


def test_Msvd_does_not_mutate_input():
    rng = np.random.default_rng(16)
    n1, n2, n3 = 4, 3, 5
    B = rng.random((n1, n2, n3))
    B_orig = B.copy()
    M = randorth(n3)
    Msvd(B, M)
    npt.assert_array_equal(B, B_orig)


def test_Msvd_non_3d_input_raises():
    B = np.zeros((4, 5))
    M = np.eye(5)
    with pytest.raises(ValueError, match="3D"):
        Msvd(B, M)


# ---------------------------------------------------------------------------
# randorth (moved from star_M to helpers)
# ---------------------------------------------------------------------------

def test_randorth_shape():
    Q = randorth(5)
    assert Q.shape == (5, 5)


def test_randorth_is_orthonormal():
    Q = randorth(7)
    npt.assert_allclose(Q.T @ Q, np.eye(7), atol=1e-12)
    npt.assert_allclose(Q @ Q.T, np.eye(7), atol=1e-12)

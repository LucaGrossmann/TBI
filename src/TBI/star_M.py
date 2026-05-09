## ----------------------------------------------------------------------------
## T-SVDM Decomposition + Helpers
## originally written by Luke Ito (2023), adapted from matlab code written by Misha
## Kilmer (2021)
## Luca Grossmann (2024) added and edited to extend to block tensor cases
## ----------------------------------------------------------------------------
import numpy as np

def mode3(A: np.ndarray, U: np.ndarray) -> np.ndarray:
    """
    Compute the mode-3 tensor-matrix product A ×_3 U.

    Parameters
    ----------
    A : np.ndarray of shape (m, p, n)
    U : np.ndarray of shape (r, n)
        Transformation matrix applied along the third mode.

    Returns
    -------
    B : np.ndarray of shape (m, p, r)
        The mode-3 product A ×_3 U.
    """
    m, p, n = A.shape
    r, q = U.shape
    if n != q:
        raise ValueError(
            f"Dimension mismatch: A.shape[2]={n} must equal U.shape[1]={q}."
        )
    A3 = np.reshape(np.transpose(A, (2, 0, 1)), (n, p * m))
    B = U @ A3
    B = np.reshape(B, (r, m, p))
    B = np.transpose(B, (1, 2, 0))
    return B


def starM(A: np.ndarray, B: np.ndarray, M: np.ndarray, vectorized: bool = True) -> np.ndarray:
    """
    Compute the ★_M tensor-tensor product C = A ★_M B.

    Transforms A and B into the M-domain via mode-3 product, multiplies
    corresponding frontal slices, then transforms back to the spatial domain.

    Parameters
    ----------
    A : np.ndarray of shape (m, p, n)
    B : np.ndarray of shape (p, l, n)
    M : np.ndarray of shape (n, n)
        Unitary (or orthogonal) transformation matrix.

    Returns
    -------
    C : np.ndarray of shape (m, l, n)
        The ★_M product A ★_M B.
    """
    m, p, n = A.shape
    q, r, s = B.shape
    n1, n2 = M.shape

    if p != q or n != s:
        raise ValueError(
            f"Dimension mismatch: A.shape {A.shape} and B.shape {B.shape} are incompatible."
        )
    if (n1, n2) != (n, n):
        raise ValueError(
            f"Dimension mismatch: Expected M of shape ({n}, {n}), got {M.shape}."
        )
    if not np.allclose(M @ M.conj().T, np.eye(n), atol=1e-8):
        raise ValueError("M is not orthogonal (or unitary).")

    Ahat = mode3(A, M)
    Bhat = mode3(B, M)
    Chat = np.zeros((m, r, n), dtype=np.result_type(A, B))

    if vectorized:
        Chat = np.einsum('mpi,pri->mri', Ahat, Bhat)
    
    else: # Easy to complete a sanity check this way. 
        for i in range(n):
            Chat[:, :, i] = Ahat[:, :, i] @ Bhat[:, :, i]

    C = mode3(Chat, M.conj().T)
    return C


def mode3_fiber(A: np.ndarray, U: np.ndarray) -> np.ndarray:
    """
    Sanity Check for mode3 above 

    Compute the mode-3 tensor-matrix product A ×_3 U via tube fibers.

    For each tube fiber A[i,j,:], computes U @ A[i,j,:]. This is equivalent
    to mode3(A, U) but follows the more interpretable definition from the
    thesis (Section 2.2): applying the linear operator U to each tube fiber.

    Parameters
    ----------
    A : np.ndarray of shape (m, p, n)
    U : np.ndarray of shape (r, n)

    Returns
    -------
    B : np.ndarray of shape (m, p, r)
    """
    m, p, n = A.shape
    r, q = U.shape
    if n != q:
        raise ValueError(
            f"Dimension mismatch: A.shape[2]={n} must equal U.shape[1]={q}."
        )
    # Loop over every (i, j) position and apply U to the tube fiber A[i,j,:]
    B = np.zeros((m, p, r))
    for i in range(m):
        for j in range(p):
            tube_fiber = A[i, j, :]        # shape (n,)
            B[i, j, :] = U @ tube_fiber    # (r, n) @ (n,) -> (r,)
    return B


def Msvd(B: np.ndarray, M: np.ndarray, compressed: bool = False):
    """
    Compute the hat-domain SVD of tensor B under the ★_M algebra.

    Computes B̂ = B ×_3 M, then takes the SVD of each frontal slice of B̂.
    Returns the hat-domain factors directly without inverting back to the
    spatial domain. Does not mutate B.

    Reconstruction per slice:
        B̂[:, :, i] ≈ U[:, :, i] @ S[:, :, i] @ V[:, :, i]

    Note: V is stored as V^H (right singular vectors as rows), following
    numpy's linalg.svd convention.

    Parameters
    ----------
    B : np.ndarray of shape (n1, n2, n3)
    M : np.ndarray of shape (n3, n3)
        Unitary transformation matrix.
    compressed : bool, default False
        If False (full SVD):
            U : (n1, n1, n3),  S : (n1, n2, n3),  V : (n2, n2, n3)
        If True (economy SVD), with k = min(n1, n2):
            U : (n1, k,  n3),  S : (k,  k,  n3),  V : (k,  n2, n3)

    Returns
    -------
    U, S, V : np.ndarray
        Hat-domain left singular vectors, diagonal singular value matrices,
        and right singular vectors (V^H).
    """
    if B.ndim != 3:
        raise ValueError(f"B must be a 3D array, got ndim={B.ndim}.")

    n1, n2, n3 = B.shape
    k = min(n1, n2)

    B_hat = mode3(B, M)  # does not mutate B

    if compressed:
        U = np.zeros((n1, k,  n3))
        S = np.zeros((k,  k,  n3))
        V = np.zeros((k,  n2, n3))
    else:
        U = np.zeros((n1, n1, n3))
        S = np.zeros((n1, n2, n3))
        V = np.zeros((n2, n2, n3))

    for i in range(n3):
        u, s, vh = np.linalg.svd(B_hat[:, :, i], full_matrices=not compressed)
        if compressed:
            U[:, :, i] = u
            S[:, :, i] = np.diag(s)
            V[:, :, i] = vh
        else:
            s_mat = np.zeros((n1, n2))
            s_mat[:k, :k] = np.diag(s)
            U[:, :, i] = u
            S[:, :, i] = s_mat
            V[:, :, i] = vh

    return U, S, V



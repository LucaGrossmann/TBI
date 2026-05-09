import numpy as np

def _block_ranges(b: np.ndarray, p: int):
    """Yield (idx, start, end) for each block defined by b."""
    k = b.shape[0]
    for i in range(k):
        start = b[i]
        end = b[i + 1] if (i + 1 < k) else p
        yield i, start, end


def _validate_inputs(X: np.ndarray, b: np.ndarray, M: np.ndarray):
    """Validate TBI inputs using helpers, raising ValueError on failure."""
    check_input(X, b, M)
    check_block(b, X.shape[1])


def _compute_total_variance(X_hat: np.ndarray) -> float:
    """Sum of squared Frobenius norms across all sheets."""
    return float(np.sum(X_hat ** 2)) 


def Mtran(A: np.ndarray) -> np.ndarray:
    """
    Compute the ★_M conjugate transpose of a tensor A.

    Transposes (and conjugates for complex tensors) each frontal slice of A.
    Equivalent to np.transpose(A, (1, 0, 2)) for real-valued tensors.

    Parameters
    ----------
    A : np.ndarray of shape (n1, n2, n3)

    Returns
    -------
    B : np.ndarray of shape (n2, n1, n3)
    """
    return np.transpose(A.conj(), (1, 0, 2))

def tensor_sum(X_k):
    """
    Compute column, row, and total sums of a 3-way tensor block.

    Used in MCIA-style normalization. For a block X_k of shape (m, p_k, n):
      - Column sum: sum over samples and sheets -> shape (1, p_k, 1)
      - Row sum: sum over variables, broadcast to (m, p_k, n)
      - Total sum: scalar sum of all elements

    Parameters
    ----------
    X_k : np.ndarray of shape (m, p_k, n)
        A single block of the tensor.

    Returns
    -------
    cs : np.ndarray of shape (1, p_k, 1)
        Column sums (summed over axis 0 and 2).
    rs : np.ndarray of shape (m, p_k, n)
        Row sums (summed over axis 1), repeated across the variable axis.
    ts : float
        Total sum of all elements.
    """
    _, p_k, _ = X_k.shape

    cs = np.sum(X_k, axis=(0, 2))
    rs = np.sum(X_k, axis=1)
    ts = np.sum(X_k)

    cs = cs.reshape(1, p_k, 1)
    rs = np.repeat(rs[:, None, :], p_k, axis=1)

    return cs, rs, ts


def check_block(b, p):
    """
    Validate block start indices for contiguous blocks over p variables.

    Parameters
    ----------
    b : array-like
        1D list/array of integer start indices. Interpreted as:
            blocks = [b[0]:b[1]], [b[1]:b[2]], ..., [b[k-1]:p]
        So you typically want b[0] == 0.
    p : int
        Number of variables (second dimension of X).

    """
    starts = np.asarray(b)

    if starts.ndim != 1:
        raise ValueError(f"`b` must be a 1D array of block start indices. Got shape {starts.shape}.")

    if starts.size == 0:
        raise ValueError("`b` must be non-empty (at least one block start index).")

    if not np.issubdtype(starts.dtype, np.integer):
        # Allow things like [0, 3, 8] that come in as float? Prefer to error loudly.
        raise TypeError(f"`b` must contain integers. Got dtype {starts.dtype}.")

    if np.any(starts < 0):
        raise ValueError("All block start indices must be >= 0.")

    if np.any(starts >= p):
        raise ValueError(f"All block start indices must be < p={p}.")

    if np.any(np.diff(starts) < 0):
        raise ValueError("Block start indices must be sorted in nondecreasing order.")

    if np.any(np.diff(starts) == 0):
        raise ValueError("Block start indices must be strictly increasing (no duplicates).")

    # Strongly recommended invariant for a partition of variables:
    if starts[0] != 0:
        raise ValueError("For a full partition of variables, `b[0]` must be 0.")

    ends = np.r_[starts[1:], p]

    # This also guarantees non-empty blocks because starts is strictly increasing
    # and ends are > starts, but we can assert for safety:
    if np.any(ends <= starts):
        raise ValueError("Invalid block definition: some block has end <= start.")

    return


def check_input(X, b, M):
    """
    Validate dimensions and basic properties of TBI inputs.

    Parameters
    ----------
    X : np.ndarray of shape (m, p, n)
        Block tensor.
    b : np.ndarray of shape (k,)
        Block start indices. Must be 1-D with at least 1 element
        and all non-negative.
    M : np.ndarray of shape (n, n)
        Square transformation matrix with n matching X.shape[2].

    Raises
    ------
    ValueError
        If any dimension or property check fails.
    """
    if len(X.shape) != 3:
        raise ValueError('X is not a 3 way tensor')
    if len(b.shape) != 1:
        raise ValueError('b is not a vector')
    if len(M.shape) != 2:
        raise ValueError('M is not a matrix')
    if X.shape[2] != M.shape[0]:
        raise ValueError('X and M have incompatible dimensions')
    if M.shape[0] != M.shape[1]:
        raise ValueError('M is not a square matrix')

    if not np.all(b >= 0):
        raise ValueError('b has negative values')
    if b.shape[0] < 1:
        raise ValueError('b must have at least one entry')


def randorth(n: int) -> np.ndarray:
    """
    Generate a random n×n orthonormal matrix via QR decomposition.

    Parameters
    ----------
    n : int
        Size of the matrix.

    Returns
    -------
    Q : np.ndarray of shape (n, n)
        Random orthonormal matrix satisfying Q.T @ Q = Q @ Q.T = I.
    """
    A = np.random.randn(n, n)
    Q, _ = np.linalg.qr(A)
    return Q

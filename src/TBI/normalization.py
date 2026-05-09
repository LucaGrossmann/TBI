import numpy as np
import copy
from .helpers import tensor_sum, check_block, _block_ranges


# ---------------------------------------------------------------------------
# Pre-built normalization pipelines for TBI, TBI II
# ---------------------------------------------------------------------------


def default_normalize(X_hat: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Variable mean-center, then block variance-normalize (tensor-wide)."""
    X_hat = variable_mean_center(X_hat, sheet_level=False)
    X_hat = variable_var_normalize(X_hat, sheet_level=False)
    
    X_hat = block_frob_normalize(X_hat, b, copy=False)
    return X_hat


def mcia_normalize(X_hat: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    CA-based normalization alternative for non-negative data.

    Offsets data to be non-negative, then applies correspondence analysis
    standardized residuals per block. Appropriate for count/frequency data
    (ecology species tables, word-frequency matrices, some genomics).
    Not recommended for mixed-scale data where the positive offset creates
    an artificial dominant component (see TBI_QA.md Q8).
    """
    return MCIA_tensor_norm(X_hat, b)


def no_normalize(X_hat: np.ndarray, b: np.ndarray) -> np.ndarray:
    """No normalization — returns a copy."""
    return X_hat.copy()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def variable_mean_center(X, sheet_level=False):
    """
    Mean-center a 3D tensor X of shape (samples, variables, timepoints).

    Parameters
    X : np.ndarray
        Array of shape (m, p, n)
    sheet_level : bool, default=False
        If True:
            Center each (variable, timepoint) across samples.
        If False:
            Center each variable globally across samples and timepoints.

    Returns
    X_centered : np.ndarray
        Mean-centered tensor of same shape as X.
    """

    if sheet_level:
        mean = np.mean(X, axis=0, keepdims=True)   # shape (1, p, n)
    else:
        mean = np.mean(X, axis=(0, 2), keepdims=True)  # shape (1, p, 1)

    return X - mean

def variable_var_normalize(X, sheet_level=False, ddof=0, eps=1e-12):
    """
    Variance-normalize (i.e., scale to unit std) a 3D tensor X of shape
    (samples, variables, timepoints), without changing the mean.

    Parameters
    X : np.ndarray
        Array of shape (m, p, n)
    sheet_level : bool, default=False
        If True:
            Normalize each (variable, timepoint) across samples -> std over axis=0
        If False:
            Normalize each variable globally across samples and timepoints -> std over axis=(0,2)
    ddof : int, default=0
        Delta degrees of freedom for std/var (0 for population, 1 for sample).
    eps : float, default=1e-12
        Any std < eps is treated as 1.0 to avoid divide-by-zero.

    Returns
    X_scaled : np.ndarray
        Variance-normalized tensor of same shape as X.
    """
    X = np.asarray(X)

    if sheet_level:
        std = np.std(X, axis=0, keepdims=True, ddof=ddof)
    else:
        std = np.std(X, axis=(0, 2), keepdims=True, ddof=ddof)

    safe_std = np.where(std < eps, 1.0, std)
    return X / safe_std


def block_var_normalize(X, b, *, sheet_level=False, ddof=0, eps=1e-12, copy=True):
    """
    Variance-normalize (scale to unit std) contiguous blocks of variables in X.

    X : np.ndarray of shape (m, p, n)
    b : 1D array-like of integer block start indices (Option A only)
        blocks are [b[0]:b[1]], ..., [b[k-1]:p]
    sheet_level : bool
        False: one std per block over axes (0,1,2) restricted to block
        True : one std per (sheet, block) over axes (variables_in_block, samples)
    ddof : int
        ddof passed to np.std
    eps : float
        if std < eps => treat std as 1.0 (leave block unchanged)
    copy : bool
        if True, returns a copy; if False, normalizes in-place

    Returns
    Y : np.ndarray
        Normalized tensor, same shape as X
    """
    X = np.asarray(X)
    if X.ndim != 3:
        raise ValueError(f"`X` must be a 3D array of shape (m, p, n). Got ndim={X.ndim}.")

    m, p, n = X.shape
    check_block(b, p)

    b = np.asarray(b, dtype=int) # ensure np.array

    Y = X.copy() if copy else X

    for _, start, end in _block_ranges(b, p):
        block = Y[:, start:end, :]  # (m, p_blk, n)

        if sheet_level: # std per sheet, shape (1, 1, n)
            std = np.std(block, axis=(0, 1), keepdims=True, ddof=ddof)

        else: # one std for entire block, shape (1, 1, 1)
            std = np.std(block, axis=(0, 1, 2), keepdims=True, ddof=ddof)

        safe_std = np.where(std < eps, 1.0, std)
        Y[:, start:end, :] = block / safe_std

    return Y

def block_frob_normalize(X, b, *, eps=1e-12, copy=True):
    """
    Frobenius-normalize contiguous blocks so each block has unit Frobenius norm.

    After normalization every block contributes equally to the total variance
    (sum of squares), regardless of block size or original scale.

    X : np.ndarray of shape (m, p, n)
    b : 1D array-like of integer block start indices
    eps : float — blocks with Frobenius norm < eps are left unchanged
    copy : bool — if True, returns a copy; if False, normalizes in-place

    Returns
    -------
    Y : np.ndarray, same shape as X
    """
    X = np.asarray(X)
    if X.ndim != 3:
        raise ValueError(f"`X` must be a 3D array of shape (m, p, n). Got ndim={X.ndim}.")

    m, p, n = X.shape
    check_block(b, p)

    b = np.asarray(b, dtype=int)

    Y = X.copy() if copy else X

    for _, start, end in _block_ranges(b, p):
        block = Y[:, start:end, :]
        fnorm = np.sqrt(np.sum(block ** 2))

        if fnorm >= eps:
            Y[:, start:end, :] = block / fnorm

    return Y


def MCIA_tensor_norm(X, b, eps=1e-12):
    """
    MCIA-style correspondence-analysis normalization for a 3D block tensor.

    For each block, computes CA standardized residuals:
        Z = (X_k - expected) / sqrt(expected)
    where expected = row_sum * col_sum / total_sum (the independence model).

    The raw residuals (X_k - expected) sum to zero per column/row by construction.
    However, dividing by sqrt(expected) breaks exact zero-sum, so the standardized
    residuals are NOT strictly mean-centered. On data requiring a large positive
    offset (e.g., mixed-scale multi-omics), the first SVD component may still
    capture a dominant near-constant pattern. Best suited for inherently non-negative
    data (counts, frequencies).

    Input data is first offset to be non-negative (CA requires non-negative data).

    Parameters
    ----------
    X : np.ndarray of shape (m, p, n)
    b : array-like of int — block start indices
    eps : float — guard against division by zero

    Returns
    -------
    Y : np.ndarray of shape (m, p, n) — CA-standardized residuals per block
    """
    b = np.asarray(b, dtype=int)
    _, p, _ = X.shape

    Y = X.copy()

    # Make values positive by offsetting by min_val (CA requires non-negative data)
    min_val = np.min(Y)
    if min_val < 0:
        Y = Y - min_val

    for _, start, end in _block_ranges(b, p):
        X_k = Y[:, start:end, :]

        cs, rs, ts = tensor_sum(X_k)

        if ts < eps:
            Y[:, start:end, :] = 0
        else:
            expected = rs * cs / ts                          # (m, p_k, n)
            safe_exp = np.where(expected < eps, eps, expected)
            Y[:, start:end, :] = (X_k - expected) / np.sqrt(safe_exp)

    return Y



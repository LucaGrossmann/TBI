"""
TBI Algorithm I — Iterative Deflation

Iterative deflation across all sheets per iteration, with configurable
normalization and energy-based stopping.

Usage
-----
    from TBI import TBI_I
    result = TBI_I(X, b, M, energy=0.95, max_iter=10)
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable, List, Optional

from .star_M import mode3
from .normalization import default_normalize, mcia_normalize, no_normalize
from .helpers import _block_ranges, _validate_inputs, _compute_total_variance





# ---------------------------------------------------------------------------
# TBIResult dataclass
# ---------------------------------------------------------------------------

@dataclass
class TBIResult:
    """Container for TBI-I decomposition results."""
    global_loadings: np.ndarray    # (p, n_iter, n)
    global_scores: np.ndarray      # (m, n_iter, n)
    local_loadings: List[np.ndarray]  # k arrays, each (p_k, n_iter, n)
    local_scores: List[np.ndarray]    # k arrays, each (m, n_iter, n)
    variance_explained: np.ndarray    # (n_iter,)
    total_variance: float
    n_iter: int


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _compute_global_loadings(X_hat: np.ndarray) -> np.ndarray:
    """
    SVD each sheet, take first right singular vector.

    Returns
    -------
    loadings : (p, n) — one loading vector per sheet.
    """
    _, p, n = X_hat.shape
    loadings = np.zeros((p, n))
    for i in range(n):
        _, _, Vt = np.linalg.svd(X_hat[:, :, i], full_matrices=False)
        loadings[:, i] = Vt[0, :]
    return loadings


def _compute_global_scores(X_hat: np.ndarray, loadings: np.ndarray) -> np.ndarray:
    """
    Project data onto global loadings: scores[:,i] = X[:,:,i] @ loadings[:,i].

    Returns
    -------
    scores : (m, n)
    """
    m, _, n = X_hat.shape
    scores = np.zeros((m, n))
    for i in range(n):
        scores[:, i] = X_hat[:, :, i] @ loadings[:, i]
    return scores


def _compute_block_loadings(
    X_hat: np.ndarray, b: np.ndarray, scores: np.ndarray
) -> List[np.ndarray]:
    """
    Block loadings: normalized projection of block data onto global scores.

    Returns
    -------
    block_loadings : list of k arrays, each (p_k, n)
    """
    _, p, n = X_hat.shape
    block_loadings = []
    for _, start, end in _block_ranges(b, p):
        p_k = end - start
        bl = np.zeros((p_k, n))
        for i in range(n):
            numerator = X_hat[:, start:end, i].T @ scores[:, i]
            denom = np.linalg.norm(numerator)
            if denom < 1e-16:
                bl[:, i] = 0.0
            else:
                bl[:, i] = numerator / denom
        block_loadings.append(bl)
    return block_loadings


def _compute_block_scores(
    X_hat: np.ndarray, b: np.ndarray, block_loadings: List[np.ndarray]
) -> List[np.ndarray]:
    """
    Block scores: project block data onto block loadings.

    Returns
    -------
    block_scores : list of k arrays, each (m, n)
    """
    m, p, n = X_hat.shape
    block_scores = []
    for (idx, start, end), bl in zip(_block_ranges(b, p), block_loadings):
        bs = np.zeros((m, n))
        for i in range(n):
            bs[:, i] = X_hat[:, start:end, i] @ bl[:, i]
        block_scores.append(bs)
    return block_scores


def _deflate(
    X_hat: np.ndarray, b: np.ndarray, block_loadings: List[np.ndarray]
) -> np.ndarray:
    """
    Deflate: remove rank-1 component from each block/sheet.

    X_k -= X_k @ a @ a.T  per block per sheet.
    """
    _, p, n = X_hat.shape
    X_def = X_hat.copy()
    for (idx, start, end), bl in zip(_block_ranges(b, p), block_loadings):
        for i in range(n):
            a = bl[:, i].reshape(-1, 1)  # (p_k, 1)
            X_def[:, start:end, i] -= X_def[:, start:end, i] @ a @ a.T
    return X_def


# ---------------------------------------------------------------------------
# Main algorithm
# ---------------------------------------------------------------------------

def TBI_I(
    X: np.ndarray,
    b: np.ndarray,
    M: np.ndarray,
    energy: float = 0.95,
    max_iter: int = 10,
    normalize_fn: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
) -> TBIResult:
    """
    TBI Algorithm I — iterative deflation across all sheets per iteration.

    Parameters
    ----------
    X : (m, p, n) block tensor
    b : 1D array of block start indices
    M : (n, n) orthogonal transformation matrix
    energy : float in (0, 1], cumulative variance threshold for early stopping
    max_iter : int, maximum number of deflation iterations
    normalize_fn : callable(X_hat, b) -> X_hat_normalized, or None for default

    Returns
    -------
    TBIResult dataclass with all decomposition outputs.
    """
    b = np.asarray(b, dtype=int)
    _validate_inputs(X, b, M)

    m, p, n = X.shape
    k = b.shape[0]

    if normalize_fn is None:
        normalize_fn = default_normalize

    # Transform to M-domain and normalize
    X_hat = mode3(X, M)
    X_hat = normalize_fn(X_hat, b)

    # Compute total variance before deflation
    total_variance = _compute_total_variance(X_hat)

    # Pre-allocate containers at max_iter (will trim later)
    gl = np.zeros((p, max_iter, n))
    gs = np.zeros((m, max_iter, n))
    ll = []
    ls = []
    for _, start, end in _block_ranges(b, p):
        p_k = end - start
        ll.append(np.zeros((p_k, max_iter, n)))
        ls.append(np.zeros((m, max_iter, n)))
    var_explained = np.zeros(max_iter)

    cumulative_energy = 0.0
    actual_iter = 0

    for it in range(max_iter):
        # Variance before this iteration's deflation
        var_before = _compute_total_variance(X_hat)

        # 1. Global loadings
        loadings = _compute_global_loadings(X_hat)
        gl[:, it, :] = loadings

        # 2. Global scores
        scores = _compute_global_scores(X_hat, loadings)
        gs[:, it, :] = scores

        # 3. Block loadings
        block_loadings = _compute_block_loadings(X_hat, b, scores)
        for idx in range(k):
            ll[idx][:, it, :] = block_loadings[idx]

        # 4. Block scores
        block_scores = _compute_block_scores(X_hat, b, block_loadings)
        for idx in range(k):
            ls[idx][:, it, :] = block_scores[idx]

        # 5. Deflate
        X_hat = _deflate(X_hat, b, block_loadings)

        # 6. Energy check
        var_after = _compute_total_variance(X_hat)
        var_removed = var_before - var_after
        var_explained[it] = var_removed
        if total_variance > 0:
            cumulative_energy += var_removed / total_variance

        actual_iter = it + 1

        if cumulative_energy >= energy:
            break

    # Trim to actual iterations
    return TBIResult(
        global_loadings=gl[:, :actual_iter, :],
        global_scores=gs[:, :actual_iter, :],
        local_loadings=[l[:, :actual_iter, :] for l in ll],
        local_scores=[l[:, :actual_iter, :] for l in ls],
        variance_explained=var_explained[:actual_iter],
        total_variance=total_variance,
        n_iter=actual_iter,
    )

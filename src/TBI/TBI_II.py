"""
TBI Algorithm II — Greedy Deflation

Greedy deflation: each iteration picks the ONE sheet with the largest leading
singular value, extracts one loading/score set from that sheet only, and
deflates only that sheet.  More storage-efficient than TBI-I when sheets have
unequal variance.

Usage
-----
    from TBI import TBI_II, TBIIResult
    result = TBI_II(X, b, M, energy=0.95, max_iter=10)
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable, List, Optional

from .star_M import mode3
from .normalization import default_normalize, mcia_normalize, no_normalize
from .helpers import _block_ranges, _validate_inputs, _compute_total_variance


# ---------------------------------------------------------------------------
# TBIIResult dataclass
# ---------------------------------------------------------------------------

@dataclass
class TBIIResult:
    """Container for TBI-II decomposition results.

    Unlike TBI-I, each iteration operates on a single sheet, so the results
    are 2D (no n-dimension).  ``sheet_indices`` records which sheet was
    selected at each iteration.
    """
    global_loadings: np.ndarray       # (p, n_iter)
    global_scores: np.ndarray         # (m, n_iter)
    local_loadings: List[np.ndarray]  # k arrays, each (p_k, n_iter)
    local_scores: List[np.ndarray]    # k arrays, each (m, n_iter)
    sheet_indices: np.ndarray         # (n_iter,) int — which sheet each iter
    variance_explained: np.ndarray    # (n_iter,)
    total_variance: float
    n_iter: int


# ---------------------------------------------------------------------------
# Internal helpers (specific to TBI-II)
# ---------------------------------------------------------------------------

def _svd_all_sheets(X_hat: np.ndarray) -> list:
    """
    Compute the SVD of every sheet and return cached results.

    Returns
    -------
    cache : list of n tuples (sigma1, Vt_row0)
            sigma1 = leading singular value, Vt_row0 = first right singular vector.
    """
    n = X_hat.shape[2]
    cache = []
    for j in range(n):
        _, S, Vt = np.linalg.svd(X_hat[:, :, j], full_matrices=False)
        cache.append((S[0], Vt[0].copy()))
    return cache


def _update_svd_cache(X_hat: np.ndarray, cache: list, sheet_idx: int) -> None:
    """Recompute the SVD for a single sheet and update the cache in-place."""
    _, S, Vt = np.linalg.svd(X_hat[:, :, sheet_idx], full_matrices=False)
    cache[sheet_idx] = (S[0], Vt[0].copy())


def _find_best_sheet(cache: list):
    """
    Pick the sheet with the largest leading singular value from the SVD cache.

    Returns
    -------
    loading   : (p,) — first right singular vector of the best sheet
    sheet_idx : int  — index of that sheet
    sigma_sq  : float — σ₁² of the best sheet
    """
    best_idx = 0
    best_sigma = cache[0][0]

    for j in range(1, len(cache)):
        if cache[j][0] > best_sigma:
            best_sigma = cache[j][0]
            best_idx = j

    return cache[best_idx][1].copy(), best_idx, best_sigma ** 2


def _compute_scores_single(X_hat: np.ndarray, loading: np.ndarray,
                           sheet_idx: int) -> np.ndarray:
    """Project one sheet onto the global loading.  Returns (m,) scores."""
    return X_hat[:, :, sheet_idx] @ loading


def _compute_block_loadings_single(
    X_hat: np.ndarray, b: np.ndarray, scores: np.ndarray, sheet_idx: int
) -> List[np.ndarray]:
    """Block loadings from a single sheet.  Returns list of k vectors (p_k,)."""
    _, p, _ = X_hat.shape
    block_loadings = []
    for _, start, end in _block_ranges(b, p):
        numerator = X_hat[:, start:end, sheet_idx].T @ scores
        denom = np.linalg.norm(numerator)
        if denom < 1e-16:
            block_loadings.append(np.zeros(end - start))
        else:
            block_loadings.append(numerator / denom)
    return block_loadings


def _compute_block_scores_single(
    X_hat: np.ndarray, b: np.ndarray, block_loadings: List[np.ndarray],
    sheet_idx: int
) -> List[np.ndarray]:
    """Block scores from a single sheet.  Returns list of k vectors (m,)."""
    _, p, _ = X_hat.shape
    block_scores = []
    for (_, start, end), bl in zip(_block_ranges(b, p), block_loadings):
        block_scores.append(X_hat[:, start:end, sheet_idx] @ bl)
    return block_scores


def _deflate_single(
    X_hat: np.ndarray, b: np.ndarray, block_loadings: List[np.ndarray],
    sheet_idx: int
) -> None:
    """Deflate a single sheet in-place: X_k[:,:,i] -= X_k[:,:,i] @ a @ a.T."""
    _, p, _ = X_hat.shape
    for (_, start, end), bl in zip(_block_ranges(b, p), block_loadings):
        a = bl.reshape(-1, 1)  # (p_k, 1)
        X_hat[:, start:end, sheet_idx] -= (
            X_hat[:, start:end, sheet_idx] @ a @ a.T
        )


# ---------------------------------------------------------------------------
# Main algorithm
# ---------------------------------------------------------------------------

def TBI_II(
    X: np.ndarray,
    b: np.ndarray,
    M: np.ndarray,
    energy: float = 0.95,
    max_iter: int = 10,
    normalize_fn: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
) -> TBIIResult:
    """
    TBI Algorithm II — greedy deflation on the best sheet per iteration.

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
    TBIIResult dataclass with all decomposition outputs.
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

    # Total variance before deflation
    total_variance = _compute_total_variance(X_hat)

    # Pre-allocate at max_iter (trim later)
    gl = np.zeros((p, max_iter))
    gs = np.zeros((m, max_iter))
    ll = []
    ls = []
    for _, start, end in _block_ranges(b, p):
        p_k = end - start
        ll.append(np.zeros((p_k, max_iter)))
        ls.append(np.zeros((m, max_iter)))
    sheet_indices = np.zeros(max_iter, dtype=int)
    var_explained = np.zeros(max_iter)

    cumulative_energy = 0.0
    actual_iter = 0

    # SVD cache: compute all sheets once, then update only the deflated sheet
    svd_cache = _svd_all_sheets(X_hat)

    for it in range(max_iter):
        # 1. Find best sheet (O(n) scan of cached σ₁ values)
        loading, sheet_idx, _ = _find_best_sheet(svd_cache)
        gl[:, it] = loading
        sheet_indices[it] = sheet_idx

        # 2. Global scores (from that sheet only)
        scores = _compute_scores_single(X_hat, loading, sheet_idx)
        gs[:, it] = scores

        # 3. Block loadings (from that sheet only)
        block_loadings = _compute_block_loadings_single(X_hat, b, scores, sheet_idx)
        for idx in range(k):
            ll[idx][:, it] = block_loadings[idx]

        # 4. Block scores (from that sheet only)
        block_scores = _compute_block_scores_single(X_hat, b, block_loadings, sheet_idx)
        for idx in range(k):
            ls[idx][:, it] = block_scores[idx]

        # 5. Deflate only that sheet, then recompute its SVD
        # Measure actual variance removed (σ₁² overestimates because block
        # loadings differ from the global SVD loading)
        var_before = np.sum(X_hat[:, :, sheet_idx] ** 2)
        _deflate_single(X_hat, b, block_loadings, sheet_idx)
        var_after = np.sum(X_hat[:, :, sheet_idx] ** 2)
        var_removed = var_before - var_after
        var_explained[it] = var_removed
        _update_svd_cache(X_hat, svd_cache, sheet_idx)

        # 6. Energy check
        if total_variance > 0:
            cumulative_energy += var_removed / total_variance

        actual_iter = it + 1

        if cumulative_energy >= energy:
            break

    # Trim to actual iterations
    return TBIIResult(
        global_loadings=gl[:, :actual_iter],
        global_scores=gs[:, :actual_iter],
        local_loadings=[l[:, :actual_iter] for l in ll],
        local_scores=[l[:, :actual_iter] for l in ls],
        sheet_indices=sheet_indices[:actual_iter],
        variance_explained=var_explained[:actual_iter],
        total_variance=total_variance,
        n_iter=actual_iter,
    )

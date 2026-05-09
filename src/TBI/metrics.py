"""
Reconstruction error metrics for publication.

Provides variance-based reconstruction error, per-block variance contribution
analysis, subspace angle computation, and re-exports rv_coefficient for
convenience.

Usage
-----
    from TBI.metrics import (
        reconstruction_error_from_variance,
        reconstruction_error_tbi_i,
        per_block_variance_contribution,
        subspace_angle,
        rv_coefficient,
    )
"""

import numpy as np
from typing import List, Optional, Union

from .helpers import _block_ranges, _compute_total_variance
from .star_M import mode3
from .analysis_utils import rv_coefficient  # re-export

__all__ = [
    "reconstruction_error_from_variance",
    "reconstruction_error_tbi_i",
    "per_block_variance_contribution",
    "subspace_angle",
    "rv_coefficient",
]


# ---------------------------------------------------------------------------
# Variance-based reconstruction error (works for any method)
# ---------------------------------------------------------------------------

def reconstruction_error_from_variance(
    variance_explained: np.ndarray,
    total_variance: float,
) -> float:
    """
    Compute relative Frobenius reconstruction error from variance statistics.

    Since deflation-based methods subtract rank-1 components from the data,
    the residual Frobenius norm satisfies:

        ||X_residual||_F^2 = total_variance - sum(variance_explained)

    so the relative reconstruction error is:

        error = sqrt(1 - sum(variance_explained) / total_variance)

    This avoids explicit reconstruction and works for TBI-I, TBI-II, Matrix
    MCIA, and any deflation-based method that tracks variance removed.

    Parameters
    ----------
    variance_explained : np.ndarray, shape (n_components,)
        Variance removed at each deflation iteration.
    total_variance : float
        Total variance of the (normalized) data before deflation.

    Returns
    -------
    error : float
        Relative Frobenius reconstruction error in [0, 1].
        Returns 1.0 if total_variance <= 0 (degenerate case).
    """
    if total_variance <= 0:
        return 1.0
    ratio = np.sum(variance_explained) / total_variance
    # Clamp to [0, 1] to guard against floating-point overshoot
    ratio = np.clip(ratio, 0.0, 1.0)
    return float(np.sqrt(1.0 - ratio))


# ---------------------------------------------------------------------------
# TBI-I explicit reconstruction error
# ---------------------------------------------------------------------------

def reconstruction_error_tbi_i(
    X: np.ndarray,
    result,
    b: np.ndarray,
    M: np.ndarray,
    normalize_fn=None,
) -> float:
    """
    Compute relative Frobenius reconstruction error for a TBI-I result.

    This uses the variance-based formula internally:

        error = sqrt(1 - sum(variance_explained) / total_variance)

    The result's ``variance_explained`` and ``total_variance`` fields
    are already computed relative to the normalized, M-transformed data,
    so no explicit reconstruction is needed.

    Parameters
    ----------
    X : np.ndarray, shape (m, p, n)
        Original block tensor (unused in computation, kept for API
        consistency and potential future explicit-reconstruction mode).
    result : TBIResult
        Output of TBI_I().
    b : np.ndarray, shape (k,)
        Block start indices (unused, kept for API consistency).
    M : np.ndarray, shape (n, n)
        Orthogonal transformation matrix (unused, kept for API consistency).
    normalize_fn : callable, optional
        Normalization function (unused, kept for API consistency).

    Returns
    -------
    error : float
        Relative Frobenius reconstruction error in [0, 1].
    """
    return reconstruction_error_from_variance(
        result.variance_explained, result.total_variance
    )


# ---------------------------------------------------------------------------
# Per-block variance contribution
# ---------------------------------------------------------------------------

def per_block_variance_contribution(
    X: np.ndarray,
    result,
    b: np.ndarray,
    M: np.ndarray,
    normalize_fn=None,
) -> np.ndarray:
    """
    Compute fraction of captured variance attributable to each block.

    For TBI-I, the deflation removes ``X_k[:,:,i] @ a @ a.T`` from each
    block k at each sheet i. The variance removed from block k at iteration
    t is ``||X_k[:,:,i] @ a||^2`` summed over sheets. This function computes
    these per-block contributions from the local scores (which are exactly
    the projections ``X_k @ a``).

    Supports both TBI-I (TBIResult) and TBI-II (TBIIResult).

    Parameters
    ----------
    X : np.ndarray, shape (m, p, n)
        Original block tensor (unused, kept for API consistency).
    result : TBIResult or TBIIResult
        Decomposition result containing local_scores and variance_explained.
    b : np.ndarray, shape (k,)
        Block start indices.
    M : np.ndarray, shape (n, n)
        Orthogonal transformation matrix (unused, kept for API consistency).
    normalize_fn : callable, optional
        Normalization function (unused, kept for API consistency).

    Returns
    -------
    fractions : np.ndarray, shape (k,)
        Fraction of total captured variance from each block.
        Sums to approximately 1.0.
    """
    b = np.asarray(b, dtype=int)
    k = b.shape[0]
    local_scores = result.local_scores  # list of k arrays

    block_variances = np.zeros(k)

    for idx in range(k):
        ls = local_scores[idx]
        # TBI-I: local_scores[k] has shape (m, n_iter, n)
        # TBI-II: local_scores[k] has shape (m, n_iter)
        # In both cases, sum of squares gives variance contributed by block k
        block_variances[idx] = np.sum(ls ** 2)

    total_captured = block_variances.sum()
    if total_captured < 1e-16:
        return np.zeros(k)
    return block_variances / total_captured


# ---------------------------------------------------------------------------
# Subspace angles
# ---------------------------------------------------------------------------

def subspace_angle(U1: np.ndarray, U2: np.ndarray) -> np.ndarray:
    """
    Compute principal angles between two subspaces.

    Given two matrices whose columns span subspaces, compute the principal
    angles via the SVD of ``Q1.T @ Q2`` where Q1, Q2 are orthonormal bases
    obtained from QR decomposition.

    Parameters
    ----------
    U1 : np.ndarray, shape (m, d1)
        First score/basis matrix. Columns span the first subspace.
    U2 : np.ndarray, shape (m, d2)
        Second score/basis matrix. Columns span the second subspace.

    Returns
    -------
    angles : np.ndarray, shape (min(d1, d2),)
        Principal angles in radians, sorted ascending.
        - Identical subspaces: all angles are 0.
        - Orthogonal subspaces: all angles are pi/2.
    """
    # Orthonormalize via QR
    Q1, _ = np.linalg.qr(U1, mode="reduced")
    Q2, _ = np.linalg.qr(U2, mode="reduced")

    # SVD of the cross-product
    _, sigma, _ = np.linalg.svd(Q1.T @ Q2, full_matrices=False)

    # Singular values are cosines of principal angles; clamp to [0, 1]
    sigma = np.clip(sigma, 0.0, 1.0)
    angles = np.arccos(sigma)

    return np.sort(angles)

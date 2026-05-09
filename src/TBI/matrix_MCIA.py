"""
Matrix MCIA — Classical Multiple Co-Inertia Analysis (2D baseline)

Unfolds the 3D tensor to a matrix, then performs block-weighted SVD
via iterative deflation.  This serves as a baseline to compare against
TBI-I and TBI-II, which exploit the full tensor structure.

The algorithm:
    1. Normalize:  apply normalize_fn(X, b) on the 3D tensor
       (default: same default_normalize as TBI-I/II).
       Note: matrix MCIA intentionally operates in the original domain --
       there is no mode-3 transform step. The tensor structure is discarded
       by unfolding.
    2. Unfold to 2D:  X_2d of shape (m, p*n)  (columns = all variable-sheet pairs)
    3. Iterative deflation:  extract one component per iteration via SVD of X_2d,
       deflate, repeat until energy threshold is met.

Usage
-----
    from TBI.matrix_MCIA import matrix_MCIA
    result = matrix_MCIA(X, b, energy=0.95, max_iter=10)
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional

from .helpers import _block_ranges, _validate_inputs, _compute_total_variance
from .normalization import default_normalize


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class MCIAResult:
    """Container for matrix MCIA decomposition results."""
    scores: np.ndarray              # (m, n_iter) — compromise scores
    loadings: np.ndarray            # (p*n, n_iter) — compromise loadings
    variance_explained: np.ndarray  # (n_iter,)
    total_variance: float
    n_iter: int


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------

def matrix_MCIA(
    X: np.ndarray,
    b: np.ndarray,
    energy: float = 0.95,
    max_iter: int = 10,
    eps: float = 1e-12,
    normalize_fn: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
) -> MCIAResult:
    """
    Classical matrix MCIA via mode-1 unfolding and block-weighted SVD.

    Parameters
    ----------
    X : (m, p, n) tensor
    b : 1D array of block start indices
    energy : cumulative variance fraction for early stopping
    max_iter : maximum number of components to extract
    eps : eigenvalue floor to avoid division by zero
    normalize_fn : callable(X_hat, b) -> X_hat_normalized, or None for default

    Returns
    -------
    MCIAResult with scores, loadings, variance_explained, total_variance, n_iter
    """
    b = np.asarray(b, dtype=int)

    if normalize_fn is None:
        normalize_fn = default_normalize

    m, p, n = X.shape

    # 1. Normalize (no mode-3 transform -- pure matrix method)
    X_norm = normalize_fn(X, b)

    # 2. Unfold to 2D: (m, p*n) — stack sheets side by side
    X_2d = X_norm.reshape(m, p * n)

    # 4. Iterative deflation
    total_var = float(np.sum(X_2d ** 2))

    scores_list = []
    loadings_list = []
    var_list = []
    cumulative = 0.0

    R = X_2d.copy()  # residual matrix

    for it in range(max_iter):
        # Leading left/right singular vectors
        U, s, Vt = np.linalg.svd(R, full_matrices=False)
        if s[0] < eps:
            break

        score = U[:, 0] * s[0]       # (m,)
        loading = Vt[0, :]            # (p*n,)
        var_captured = s[0] ** 2

        scores_list.append(score)
        loadings_list.append(loading)
        var_list.append(var_captured)

        cumulative += var_captured
        if total_var > eps and cumulative / total_var >= energy:
            break

        # Deflate
        R -= np.outer(score, loading)

    n_iter = len(var_list)

    return MCIAResult(
        scores=np.column_stack(scores_list) if n_iter > 0 else np.empty((m, 0)),
        loadings=np.column_stack(loadings_list) if n_iter > 0 else np.empty((p * n, 0)),
        variance_explained=np.array(var_list),
        total_variance=total_var,
        n_iter=n_iter,
    )

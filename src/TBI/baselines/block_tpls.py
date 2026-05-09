"""
block.tPLS — Block Tensor Partial Least Squares (Kodikara et al. 2026).

Implements the CANONICAL MODE of block.tPLS: unsupervised multi-block RGCCA
in the t-SVDM transform domain. For each component, solves the RGCCA criterion
per frontal slice, selects the slice with the largest criterion value, and
extracts loadings/scores from that slice.

Reference: Kodikara, Lu, Wang & Le Cao (2026), "tensorOmics: Data integration
for longitudinal omics data using tensor factorisation", bioRxiv 2026.02.10.705179.

Algorithm (paper Section 2.2, 2.3.4, Eq. 13, Algorithm 2):
1. MDF centering: *X_{:,:,k} := X_{:,:,k} - X_bar  (Eq. 3)
2. Mode-3 transform: *X_hat = *X x_3 M  (Eq. 4, default M = scaled DCT)
3. For each component h:
   a. Per frontal slice k: RGCCA NIPALS finds loading vectors a^(q)
      maximizing sum c_{f,g} cov(X_hat^(f) a^(f), X_hat^(g) a^(g))  (Eq. 13)
   b. Select slice k* with maximum criterion value
   c. Scores: b^(q) = X_hat^(q)_{:,:,k*} a^(q)
   d. Regression coefficients: c^(q) = X_hat^(q)^T b^(q) / (b^T b)  (Eq. 7)
   e. Deflation: subtract prediction from MDF data  (Eq. 8-9 adapted for Q blocks)

Deviations from paper / R reference implementation:
- Canonical mode only (no regression mode / Y-block, see Appendix A Eq. 15)
- Horst scheme only (R supports factorial, centroid schemes)
- No shrinkage regularization (R supports tau parameter; at default tau=1, equivalent)
- Slice selection via full RGCCA per slice (R uses SVD-based heuristic, faster)
- Convergence uses ||delta|| < tol (R uses ||delta||^2 < tol)

Non-original additions (not in paper):
- Global scores via mean across block scores (paper defines only per-block scores)
- Variance tracking via Frobenius norm reduction (paper does not define this for PLS)
- Early stopping when criterion <= 0
- SVD-based loading initialization (paper does not specify)

Dependencies: numpy only (no external packages required).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class BlockTPLSResult:
    """Container for block.tPLS decomposition results."""
    scores: np.ndarray                          # (m, n_components) global scores
    block_scores: List[np.ndarray] = field(repr=False)   # per-block (m, n_components)
    loadings: List[np.ndarray] = field(repr=False)       # per-block (p_q, n_components)
    variance_explained: np.ndarray = field(repr=False)    # (n_components,)
    total_variance: float = 0.0
    n_iter: int = 0
    sheet_indices: np.ndarray = field(repr=False, default_factory=lambda: np.array([], dtype=int))


# ---------------------------------------------------------------------------
# Internal helpers — own implementations, NOT shared with TBI core
# ---------------------------------------------------------------------------

def _split_blocks(X, b):
    """Split tensor (m, p, n) into Q sub-tensors along the variable axis.

    Parameters
    ----------
    X : (m, p, n) array
    b : 1D int array of block start indices (first element must be 0)

    Returns
    -------
    list of Q arrays, each (m, p_q, n)
    """
    m, p, n = X.shape
    Q = len(b)
    blocks = []
    for i in range(Q):
        start = int(b[i])
        end = int(b[i + 1]) if i + 1 < Q else p
        blocks.append(X[:, start:end, :].copy())
    return blocks


def _mdf_center(X):
    """Mean Deviation Form centering (Eq. 3 of Kodikara et al. 2026).

    Subtracts the cohort-average trajectory from each subject:
        *X_{:,:,k} := X_{:,:,k} - X_bar,
        where X_bar = (1/n) sum_i X_{i,:,:}

    Parameters
    ----------
    X : (m, p, n) array

    Returns
    -------
    (m, p, n) centered array
    """
    X_bar = X.mean(axis=0, keepdims=True)  # (1, p, n)
    return X - X_bar


def _scaled_dct(t):
    """Scaled DCT-II matrix of size (t, t).

    Matches the definition in Mor et al. (2022) / Kodikara et al. (2026).
    """
    D = np.zeros((t, t))
    for k in range(t):
        for j in range(t):
            if k == 0:
                D[k, j] = 1.0 / np.sqrt(t)
            else:
                D[k, j] = np.sqrt(2.0 / t) * np.cos(
                    np.pi * (2 * j + 1) * k / (2 * t)
                )
    return D


def _mode3_transform(X, M):
    """Apply mode-3 product: X_hat[:,:,k] = sum_j M[k,j] X[:,:,j].

    Parameters
    ----------
    X : (m, p, n) array
    M : (n, n) invertible transform matrix
    """
    return np.einsum('mpj,kj->mpk', X, M)


def _rgcca_nipals_slice(blocks_k, C, tol=1e-6, max_inner=100, rng=None):
    """RGCCA NIPALS for one frontal slice (Tenenhaus & Tenenhaus 2011).

    Solves: max sum_{f!=g} c_{f,g} cov(X^(f) a^(f), X^(g) a^(g))
            s.t. ||a^(q)||_2 = 1  for all q

    The NIPALS update for block q is (Horst scheme):
        z_q = sum_{f!=q} c_{q,f} * X_q^T X_f a_f
        a_q = z_q / ||z_q||

    Parameters
    ----------
    blocks_k : list of Q arrays, each (m, p_q) — one frontal slice per block
    C : (Q, Q) symmetric design matrix with zeros on diagonal
    tol : convergence tolerance on max loading change
    max_inner : max NIPALS iterations

    Returns
    -------
    loadings : list of Q unit vectors, each (p_q,)
    criterion : scalar — sum of pairwise weighted covariances
    n_inner : int — number of NIPALS iterations used
    """
    Q = len(blocks_k)
    if rng is None:
        rng = np.random.default_rng(0)

    # ! NON-ORIGINAL: SVD-based initialization (paper does not specify;
    # R uses SVD + optional shrinkage normalization, equivalent at tau=1)
    loadings = []
    for q in range(Q):
        if blocks_k[q].shape[1] == 0:
            loadings.append(np.array([]))
            continue
        _, _, Vt = np.linalg.svd(blocks_k[q], full_matrices=False)
        loadings.append(Vt[0, :].copy())

    n_inner = 0
    for it in range(max_inner):
        n_inner = it + 1
        max_change = 0.0

        for q in range(Q):
            if blocks_k[q].shape[1] == 0:
                continue

            # Gradient: z = sum_{f!=q} c_{q,f} * X_q^T X_f a_f
            z = np.zeros(blocks_k[q].shape[1])
            for f in range(Q):
                if f != q and C[q, f] != 0 and blocks_k[f].shape[1] > 0:
                    z += C[q, f] * (blocks_k[q].T @ blocks_k[f] @ loadings[f])

            norm_z = np.linalg.norm(z)
            if norm_z > 1e-15:
                a_new = z / norm_z
                max_change = max(max_change, np.linalg.norm(a_new - loadings[q]))
                loadings[q] = a_new

        if max_change < tol:
            break

    # Criterion: sum_{f<g} c_{f,g} * a_f^T X_f^T X_g a_g
    criterion = 0.0
    for f in range(Q):
        for g in range(f + 1, Q):
            if C[f, g] != 0 and blocks_k[f].shape[1] > 0 and blocks_k[g].shape[1] > 0:
                score_f = blocks_k[f] @ loadings[f]
                score_g = blocks_k[g] @ loadings[g]
                criterion += C[f, g] * np.dot(score_f, score_g)

    return loadings, criterion, n_inner


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def block_tpls(
    X: np.ndarray,
    b: np.ndarray,
    M: Optional[np.ndarray] = None,
    C: Optional[np.ndarray] = None,
    n_components: int = 10,
    use_mdf: bool = True,
    tol: float = 1e-6,
    max_inner: int = 100,
) -> BlockTPLSResult:
    """
    Block Tensor PLS via RGCCA NIPALS in the transform domain.

    For each component, the RGCCA criterion (Eq. 13) is solved per frontal
    slice of the transformed tensor. The slice with the largest criterion
    value is selected, and loadings/scores are extracted from that slice.
    Deflation follows Algorithm 2 (Eq. 8-9), adapted for Q blocks.

    Parameters
    ----------
    X : (m, p, n) tensor — m subjects, p variables, n timepoints/sheets
    b : 1D array of block start indices partitioning the p variables
    M : (n, n) invertible transform matrix. If None, uses scaled DCT.
    C : (Q, Q) symmetric design matrix specifying block connections.
        If None, uses fully connected: C[f,g] = 1 for f != g.
    n_components : maximum number of components to extract
    use_mdf : if True, apply Mean Deviation Form centering (Eq. 3)
    tol : NIPALS convergence tolerance
    max_inner : max NIPALS iterations per slice per component

    Returns
    -------
    BlockTPLSResult
    """
    b = np.asarray(b, dtype=int)
    m, p, n = X.shape
    Q = len(b)

    if C is None:
        C = np.ones((Q, Q)) - np.eye(Q)
    else:
        C = np.asarray(C, dtype=float)

    if M is None:
        M = _scaled_dct(n)
    M_inv = np.linalg.inv(M)

    # Split into blocks
    blocks_mdf = _split_blocks(X, b)

    # MDF centering (Eq. 3)
    if use_mdf:
        blocks_mdf = [_mdf_center(blk) for blk in blocks_mdf]

    # Total variance (after centering, before deflation)
    total_var = sum(float(np.sum(blk ** 2)) for blk in blocks_mdf)

    # Storage
    all_block_scores = [[] for _ in range(Q)]
    all_block_loadings = [[] for _ in range(Q)]
    var_list = []
    sheet_list = []

    for h in range(n_components):
        # Mode-3 transform (Eq. 4) — re-applied each iteration because
        # deflation is in MDF domain (Algorithm 2, lines 15-16).
        # Since the transform is linear, this is equivalent to deflating
        # in the transform domain, but we follow the paper's structure.
        blocks_hat = [_mode3_transform(blk, M) for blk in blocks_mdf]

        # Per-slice RGCCA NIPALS: find the slice k* maximizing the criterion
        best_criterion = -np.inf
        best_k = -1
        best_loadings = None

        for k in range(n):
            slices_k = [blk_hat[:, :, k] for blk_hat in blocks_hat]

            # Skip if any block slice is effectively zero
            if any(np.linalg.norm(s) < 1e-15 for s in slices_k):
                continue

            loadings_k, criterion_k, _ = _rgcca_nipals_slice(
                slices_k, C, tol=tol, max_inner=max_inner
            )

            if criterion_k > best_criterion:
                best_criterion = criterion_k
                best_k = k
                best_loadings = loadings_k

        # ! NON-ORIGINAL: early stopping when criterion <= 0
        if best_loadings is None or best_criterion <= 0:
            break

        sheet_list.append(best_k)

        # Scores: b^(q) = X_hat^(q)_{:,:,k*} a^(q)  (Algorithm 2, lines 6/8)
        scores_h = []
        for q in range(Q):
            b_q = blocks_hat[q][:, :, best_k] @ best_loadings[q]
            scores_h.append(b_q)
            all_block_scores[q].append(b_q)
            all_block_loadings[q].append(best_loadings[q])

        # ! NON-ORIGINAL: variance tracking via norm reduction (paper does not
        # define variance explained for PLS methods)
        norm_before = sum(np.sum(blk ** 2) for blk in blocks_mdf)

        # Deflation (Algorithm 2, lines 9-16, adapted for Q blocks)
        for q in range(Q):
            b_q = scores_h[q]
            btb = np.dot(b_q, b_q)
            if btb < 1e-15:
                continue

            # Regression coefficient (Eq. 7): c = X_hat^T b / (b^T b)
            c_q = blocks_hat[q][:, :, best_k].T @ b_q / btb

            # Prediction: nonzero only in slice k* (Algorithm 2, line 11)
            pred_hat = np.zeros_like(blocks_hat[q])
            pred_hat[:, :, best_k] = np.outer(b_q, c_q)

            # Transform back to MDF domain (Algorithm 2, line 13)
            pred_mdf = _mode3_transform(pred_hat, M_inv)

            # Deflate in MDF domain (Algorithm 2, lines 15-16)
            blocks_mdf[q] = blocks_mdf[q] - pred_mdf

        norm_after = sum(np.sum(blk ** 2) for blk in blocks_mdf)
        var_list.append(max(0.0, norm_before - norm_after))

    # Assemble results
    n_extracted = len(var_list)

    if n_extracted > 0:
        block_scores_out = []
        block_loadings_out = []
        for q in range(Q):
            block_scores_out.append(np.column_stack(all_block_scores[q]))
            block_loadings_out.append(np.column_stack(all_block_loadings[q]))

        # ! NON-ORIGINAL: global scores via block averaging (paper defines
        # only per-block scores; this is for BenchmarkResult compatibility)
        global_scores = np.mean(
            [bs for bs in block_scores_out], axis=0
        )
    else:
        global_scores = np.empty((m, 0))
        block_scores_out = [np.empty((m, 0)) for _ in range(Q)]
        block_loadings_out = [
            np.empty((blocks_mdf[q].shape[1], 0)) for q in range(Q)
        ]

    return BlockTPLSResult(
        scores=global_scores,
        block_scores=block_scores_out,
        loadings=block_loadings_out,
        variance_explained=np.array(var_list),
        total_variance=total_var,
        n_iter=n_extracted,
        sheet_indices=np.array(sheet_list, dtype=int),
    )

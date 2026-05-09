"""
Shared analysis utilities — plotting, metrics, and tensor helpers.

Used by bio_demo.py, cmipb_pipeline.py, and tcam_comparison.py.
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from .helpers import _block_ranges

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DCT matrix
# ---------------------------------------------------------------------------

def dct_matrix(N: int) -> np.ndarray:
    """Orthonormal Type-II DCT matrix of size N x N."""
    i = np.arange(N).reshape((N, 1))
    j = np.arange(N)
    D = np.sqrt(2.0 / N) * np.cos(np.pi * (2 * j + 1) * i / (2 * N))
    D[0, :] = np.sqrt(1.0 / N)
    return D


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def rv_coefficient(X: np.ndarray, Y: np.ndarray) -> float:
    """
    RV coefficient between two (m, d) score matrices.

    Measures multivariate correlation between configurations.
    Returns a value in [0, 1].
    """
    XX = X @ X.T
    YY = Y @ Y.T
    num = np.trace(XX @ YY)
    denom = np.sqrt(np.trace(XX @ XX) * np.trace(YY @ YY))
    if denom < 1e-16:
        return 0.0
    return float(num / denom)


def block_variance_contributions(
    X_hat: np.ndarray, b: np.ndarray
) -> np.ndarray:
    """
    Compute fraction of total variance in each block.

    Parameters
    ----------
    X_hat : (m, p, n) tensor (typically in transform domain)
    b : block start indices

    Returns
    -------
    fractions : (k,) array summing to 1
    """
    _, p, _ = X_hat.shape
    b = np.asarray(b, dtype=int)
    variances = np.zeros(len(b))
    for idx, start, end in _block_ranges(b, p):
        variances[idx] = np.sum(X_hat[:, start:end, :] ** 2)
    total = variances.sum()
    if total < 1e-16:
        return variances
    return variances / total


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def scree_plot(
    results_dict: Dict[str, np.ndarray],
    total_variances: Optional[Dict[str, float]] = None,
    title: str = "Scree Plot",
    savepath: Optional[str] = None,
):
    """
    Overlay cumulative variance-explained curves for multiple methods.

    Parameters
    ----------
    results_dict : {"method_name": variance_explained_array}
    total_variances : {"method_name": total_variance} for computing proportions.
                      If None, plots raw variance values.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: per-component
    ax = axes[0]
    for name, var in results_dict.items():
        if total_variances and name in total_variances:
            var = var / total_variances[name]
        ax.bar(np.arange(len(var)) + 1, var, alpha=0.5, label=name)
    ax.set_xlabel("Component")
    ax.set_ylabel("Variance Proportion")
    ax.set_title(f"{title} — Per Component")
    ax.legend()

    # Right: cumulative
    ax = axes[1]
    for name, var in results_dict.items():
        if total_variances and name in total_variances:
            var = var / total_variances[name]
        cumulative = np.cumsum(var)
        ax.plot(np.arange(len(cumulative)) + 1, cumulative, marker="o",
                label=name)
    ax.set_xlabel("Component")
    ax.set_ylabel("Cumulative Variance Proportion")
    ax.set_title(f"{title} — Cumulative")
    ax.legend()

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {savepath}")
    plt.close()


def score_scatter(
    scores: np.ndarray,
    labels: Optional[np.ndarray] = None,
    label_names: Optional[Dict] = None,
    title: str = "Score Plot",
    xlabel: str = "Component 1",
    ylabel: str = "Component 2",
    savepath: Optional[str] = None,
):
    """
    2D scatter of first two components, optionally colored by group labels.

    Parameters
    ----------
    scores : (m, d) array — at least 2 columns
    labels : (m,) array of group indices (ints)
    label_names : {int: "group name"} for legend
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    if labels is not None:
        unique_labels = np.unique(labels)
        for lbl in unique_labels:
            mask = labels == lbl
            name = label_names[lbl] if label_names else str(lbl)
            ax.scatter(scores[mask, 0], scores[mask, 1], label=name, alpha=0.7)
        ax.legend()
    else:
        ax.scatter(scores[:, 0], scores[:, 1], alpha=0.7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {savepath}")
    plt.close()


def score_scatter_grid(
    scores_dict: Dict[str, np.ndarray],
    labels: Optional[np.ndarray] = None,
    label_names: Optional[Dict] = None,
    title: str = "Score Comparison",
    ncols: int = 3,
    savepath: Optional[str] = None,
):
    """
    Subplot grid of 2D score scatter plots for multiple methods.

    Parameters
    ----------
    scores_dict : {"method_name": (m, d) array} — at least 2 columns each
    labels : (m,) array of group indices for coloring
    label_names : {int: "group name"} for legend
    ncols : number of columns in the grid
    """
    methods = list(scores_dict.keys())
    n_methods = len(methods)
    if n_methods == 0:
        return

    ncols = min(ncols, n_methods)
    nrows = int(np.ceil(n_methods / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 4.5 * nrows),
                             squeeze=False)

    handles, legend_labels = None, None

    for idx, method_name in enumerate(methods):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        scores = scores_dict[method_name]

        if labels is not None:
            unique_labels = np.unique(labels)
            for lbl in unique_labels:
                mask = labels == lbl
                name = label_names[lbl] if label_names else str(lbl)
                ax.scatter(scores[mask, 0], scores[mask, 1],
                           label=name, alpha=0.7, s=20)
            if handles is None:
                handles, legend_labels = ax.get_legend_handles_labels()
        else:
            ax.scatter(scores[:, 0], scores[:, 1], alpha=0.7, s=20)

        ax.set_title(method_name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Component 1", fontsize=9)
        ax.set_ylabel("Component 2", fontsize=9)
        ax.tick_params(labelsize=8)

    # Hide unused subplots
    for idx in range(n_methods, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].axis("off")

    # Shared legend at bottom
    if handles is not None:
        fig.legend(handles, legend_labels, loc="lower center",
                   ncol=len(legend_labels), fontsize=10,
                   bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0.03 if handles else 0, 1, 0.95])

    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {savepath}")
    plt.close()


def top_loadings_table(
    loadings: np.ndarray,
    variable_names: List[str],
    k: int = 10,
    component: int = 0,
) -> list:
    """
    Return the top-k variables by absolute loading value for a given component.

    Parameters
    ----------
    loadings : (p,) or (p, n_iter) or (p, n_iter, n)
    variable_names : list of p variable names
    k : number of top variables to return
    component : which component index

    Returns
    -------
    List of (variable_name, loading_value) tuples, sorted by |loading|.
    """
    if loadings.ndim == 1:
        vals = loadings
    elif loadings.ndim == 2:
        vals = loadings[:, component]
    else:
        # 3D: sum absolute loadings across sheets for ranking
        vals = np.sum(np.abs(loadings[:, component, :]), axis=-1)

    top_idx = np.argsort(np.abs(vals))[::-1][:k]
    return [(variable_names[i], float(vals[i])) for i in top_idx]


def efficiency_plot(
    results_dict: Dict[str, np.ndarray],
    total_variances: Dict[str, float],
    storage_per_component: Dict[str, int],
    title: str = "Storage Efficiency",
    savepath: Optional[str] = None,
):
    """
    Plot cumulative variance explained vs cumulative stored values
    (scores + loadings per component).

    Parameters
    ----------
    results_dict : {"method_name": variance_explained_array}
    total_variances : {"method_name": total_variance}
    storage_per_component : {"method_name": total_values_stored_per_component}
        Scores + loadings.  e.g. TBI-I stores (m+p)*n, TBI-II stores m+p,
        Matrix MCIA stores m + p*n.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, var in results_dict.items():
        total_var = total_variances[name]
        size_per_comp = storage_per_component[name]

        cum_var = np.cumsum(var) / total_var * 100
        cum_storage = np.arange(1, len(var) + 1) * size_per_comp

        ax.plot(cum_storage, cum_var, marker="o", markersize=4, label=name)

    ax.set_xlabel("Stored Values per Component (Scores + Loadings)")
    ax.set_ylabel("Cumulative Variance Explained (%)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {savepath}")
    plt.close()


def block_contribution_bar(
    variance_fractions: np.ndarray,
    block_names: List[str],
    title: str = "Block Variance Contribution",
    savepath: Optional[str] = None,
):
    """
    Bar chart showing variance fraction per block.

    Parameters
    ----------
    variance_fractions : (k,) array from block_variance_contributions()
    block_names : list of k block names
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(block_names))
    ax.bar(x, variance_fractions, color="steelblue", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(block_names, rotation=45, ha="right")
    ax.set_ylabel("Fraction of Total Variance")
    ax.set_title(title)
    ax.axhline(1.0 / len(block_names), color="red", linestyle="--",
               label=f"Equal share ({1.0/len(block_names):.2%})")
    ax.legend()

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {savepath}")
    plt.close()


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------

_TENSOR_SHEET_METHODS = {"TBI-I", "TCAM"}
_GREEDY_METHODS = {"TBI-II", "block-tPLS"}
# Everything else (matrix methods): m + p*n


def storage_per_component(method_name: str, m: int, p: int, n: int) -> int:
    """
    Number of stored values (scores + loadings) per component for a method.

    - Tensor-sheet methods (TBI-I, TCAM): (m+p)*n
    - Greedy methods (TBI-II, block-tPLS): m+p
      block-tPLS selects one slice k* per component (like TBI-II), so it stores
      one loading (p) + one score (m) per component, not all n sheets.
    - Matrix methods (MCIA, MFA, STATIS, MOFA, etc.): m + p*n
    """
    if method_name in _TENSOR_SHEET_METHODS:
        return (m + p) * n
    elif method_name in _GREEDY_METHODS:
        return m + p
    else:
        return m + p * n

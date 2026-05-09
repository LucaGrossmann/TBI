"""
Clustering benchmark: ARI on controlled simulations with planted group structure.

Generates synthetic block tensors where group membership is encoded in
temporally-modulated signals — the kind of structure that tensor methods
(via mode-3 transform) can exploit while matrix methods lose by unfolding.
Sweeps signal-to-noise ratio (SNR) to show the threshold where tensor methods
start separating groups that matrix methods miss.

Literature justification:
- Cantini et al. (2021, Nature Communications 12:124): Controlled simulations
  via InterSIM with planted group labels, evaluated by ARI, as the primary
  benchmark for 9 multi-omics integration methods.
- Rappoport & Shamir (2018, Nucleic Acids Research 46(20):10546-10562): ARI
  for evaluating multi-omics clustering across 10 TCGA cancer types.
- Hubert & Arabie (1985, Journal of Classification 2(1):193-218): ARI is
  adjusted for chance (E[ARI] = 0 for random clustering), making it directly
  interpretable.

Simulation design:
- Two groups of subjects, each with a distinct temporal profile across sheets.
- Group signal is rank-2, shared across blocks but with block-specific weights.
- After DCT transform, temporal patterns concentrate into fewer sheets —
  tensor methods exploit this; matrix unfolding dilutes it.
- Sweep SNR in {0.5, 2.0, 5.0, 10.0} with 10 seeds per level.

Output
------
Single figure (figures/benchmarks/clustering_ari.png) with 4 subplots
(one per SNR level). Each subplot shows ARI vs. number of components
(mean line + shaded std band). Higher is better.

Usage
-----
    conda run -n claude python benchmarks/clustering_benchmark.py
"""

import os
import time
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from TBI import TBI_I, TBI_II, matrix_MCIA
from TBI.analysis_utils import dct_matrix
from TBI.result_types import adapt_tbi_i, adapt_tbi_ii, adapt_mcia, BenchmarkResult

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures", "benchmarks")
ENERGY = 0.99
MAX_ITER = 15
MAX_COMPONENTS_EVAL = 10
N_SEEDS = 10

# Simulation parameters
M_SUBJECTS = 60       # 30 per group
P_VARIABLES = 100     # total variables
N_SHEETS = 6          # timepoints
BLOCK_SIZES = [40, 35, 25]  # 3 blocks
N_GROUPS = 2
SNR_LEVELS = [0.5, 2.0, 5.0, 10.0]


# ---------------------------------------------------------------------------
# Simulation: synthetic block tensor with planted temporal group structure
# ---------------------------------------------------------------------------

def generate_simulation(snr, rng):
    """Generate a synthetic (m, p, n) block tensor with planted group structure.

    The group signal is a rank-2 component where:
    - Component 1: groups differ in temporal profile (group 1 peaks early,
      group 2 peaks late). This is the structure tensor methods exploit.
    - Component 2: groups differ in block contribution (group 1 strong in
      block 1, group 2 strong in block 3).

    After DCT transform, the smooth temporal patterns concentrate into the
    first few DCT coefficients, making them easier for TBI to extract.
    Matrix MCIA unfolds the tensor and dilutes this temporal structure.

    Parameters
    ----------
    snr : float
        Signal-to-noise ratio. Signal power is fixed; noise std = 1/snr.
    rng : np.random.Generator

    Returns
    -------
    X : (m, p, n) array
    b : (k,) block start indices
    labels : (m,) integer group labels (0 or 1)
    """
    m, p, n = M_SUBJECTS, P_VARIABLES, N_SHEETS
    k = len(BLOCK_SIZES)
    b = np.array([0] + list(np.cumsum(BLOCK_SIZES[:-1])), dtype=int)

    # Group labels: first half = group 0, second half = group 1
    labels = np.array([0] * (m // 2) + [1] * (m - m // 2))

    # Temporal profiles: smooth functions that differ between groups
    t = np.linspace(0, np.pi, n)
    temporal_g0 = np.sin(t)          # peaks in the middle
    temporal_g1 = np.cos(t / 2)      # peaks early, decays

    # Normalize temporal profiles to unit norm
    temporal_g0 /= np.linalg.norm(temporal_g0)
    temporal_g1 /= np.linalg.norm(temporal_g1)

    # Block-specific signal weights (how much each block contributes)
    block_weights_c1 = [1.0, 0.7, 0.5]   # component 1: all blocks contribute
    block_weights_c2 = [0.3, 0.5, 1.0]   # component 2: block 3 dominant

    # Generate fixed loading vectors per block (shared across all subjects)
    loadings_c1 = []
    loadings_c2 = []
    for ki in range(k):
        start = b[ki]
        end = b[ki + 1] if ki + 1 < k else p
        p_k = end - start
        v1 = rng.standard_normal(p_k)
        v1 /= np.linalg.norm(v1)
        v2 = rng.standard_normal(p_k)
        # Orthogonalize component 2 against component 1
        v2 -= (v2 @ v1) * v1
        v2 /= np.linalg.norm(v2)
        loadings_c1.append(v1)
        loadings_c2.append(v2)

    # Generate signal: group membership determines temporal profile
    X = np.zeros((m, p, n))

    for i in range(m):
        g = labels[i]
        # Subject-specific score (random magnitude around group mean)
        score_c1 = 1.0 + 0.3 * rng.standard_normal()
        score_c2 = 0.7 + 0.3 * rng.standard_normal()

        temp_c1 = temporal_g0 if g == 0 else temporal_g1
        temp_c2 = temporal_g1 if g == 0 else temporal_g0

        for ki in range(k):
            start = b[ki]
            end = b[ki + 1] if ki + 1 < k else p

            signal = (score_c1 * block_weights_c1[ki]
                      * np.outer(loadings_c1[ki], temp_c1))
            signal += (score_c2 * block_weights_c2[ki]
                       * np.outer(loadings_c2[ki], temp_c2))
            X[i, start:end, :] = signal

    # Add noise
    noise_std = 1.0 / snr
    X += noise_std * rng.standard_normal(X.shape)

    return X, b, labels


# ---------------------------------------------------------------------------
# Method runners
# ---------------------------------------------------------------------------

def run_tbi_i(X, b, M):
    result = TBI_I(X, b, M, energy=ENERGY, max_iter=MAX_ITER)
    # Average scores across all sheets for clustering (sheet 0 alone may
    # miss the group signal if it concentrates on other DCT coefficients)
    scores_avg = result.global_scores[:, :result.n_iter, :].mean(axis=2)
    return BenchmarkResult(
        method_name="TBI-I",
        scores=scores_avg,
        variance_explained=result.variance_explained,
        total_variance=result.total_variance,
        n_iter=result.n_iter,
    )


def run_tbi_ii(X, b, M):
    n = X.shape[2]
    result = TBI_II(X, b, M, energy=ENERGY, max_iter=MAX_ITER * n)
    return adapt_tbi_ii(result, elapsed=0.0)


def run_matrix_mcia(X, b, M):
    result = matrix_MCIA(X, b, energy=ENERGY, max_iter=MAX_ITER)
    return adapt_mcia(result, elapsed=0.0)


def run_block_tpls(X, b, M):
    try:
        from TBI.baselines.block_tpls import block_tpls
    except ImportError:
        return None
    result = block_tpls(X, b, M=M, n_components=MAX_ITER)
    return BenchmarkResult(
        method_name="block-tPLS",
        scores=result.scores,
        variance_explained=result.variance_explained,
        total_variance=result.total_variance,
        n_iter=result.n_iter,
    )


METHODS = [
    ("TBI-I", run_tbi_i, True),
    ("TBI-II", run_tbi_ii, True),
    ("Matrix MCIA", run_matrix_mcia, False),
    ("block-tPLS", run_block_tpls, True),
]


# ---------------------------------------------------------------------------
# ARI computation
# ---------------------------------------------------------------------------

def compute_ari_curve(scores, labels, n_groups, max_d=MAX_COMPONENTS_EVAL):
    """Compute ARI for d = 1..max_d components."""
    n_available = scores.shape[1]
    max_d = min(max_d, n_available)
    if max_d == 0:
        return np.array([]), np.array([])

    dims = np.arange(1, max_d + 1)
    aris = np.zeros(max_d)

    for i, d in enumerate(dims):
        S = scores[:, :d]
        km = KMeans(n_clusters=n_groups, n_init=50, random_state=42)
        pred = km.fit_predict(S)
        aris[i] = adjusted_rand_score(labels, pred)

    return dims, aris


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    n_snr = len(SNR_LEVELS)
    fig, axes = plt.subplots(1, n_snr, figsize=(5 * n_snr, 4.5), squeeze=False)
    axes = axes[0]

    colors = {"TBI-I": "#1f77b4", "TBI-II": "#ff7f0e",
              "Matrix MCIA": "#2ca02c", "block-tPLS": "#d62728"}

    for col, snr in enumerate(SNR_LEVELS):
        ax = axes[col]
        print(f"\n{'='*50}")
        print(f"SNR = {snr}")

        # Collect ARI curves across seeds for each method
        method_aris = {name: [] for name, _, _ in METHODS}

        for seed in range(N_SEEDS):
            rng = np.random.default_rng(seed)
            X, b, labels = generate_simulation(snr, rng)
            m, p, n = X.shape
            M = dct_matrix(n)

            for method_name, runner, requires_tensor in METHODS:
                if requires_tensor and n < 2:
                    continue
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        result = runner(X, b, M)
                    if result is None:
                        continue
                    dims, aris = compute_ari_curve(
                        result.scores, labels, N_GROUPS)
                    if len(aris) > 0:
                        method_aris[method_name].append(aris)
                except Exception:
                    pass

        # Plot mean ± std for each method
        for method_name, _, _ in METHODS:
            ari_list = method_aris[method_name]
            if not ari_list:
                continue

            # Pad to same length
            max_len = max(len(a) for a in ari_list)
            padded = np.full((len(ari_list), max_len), np.nan)
            for i, a in enumerate(ari_list):
                padded[i, :len(a)] = a

            dims = np.arange(1, max_len + 1)
            mean_ari = np.nanmean(padded, axis=0)
            std_ari = np.nanstd(padded, axis=0)

            c = colors.get(method_name, "gray")
            ax.plot(dims, mean_ari, "o-", label=method_name,
                    color=c, markersize=3, linewidth=1.5)
            ax.fill_between(dims, mean_ari - std_ari, mean_ari + std_ari,
                            alpha=0.15, color=c)

            best_idx = np.nanargmax(mean_ari)
            print(f"  {method_name:<15s}  mean_ARI={mean_ari[best_idx]:.3f} "
                  f"± {std_ari[best_idx]:.3f} at d={dims[best_idx]}")

        ax.set_xlabel("Number of Components")
        if col == 0:
            ax.set_ylabel("Adjusted Rand Index")
        ax.set_title(f"SNR = {snr}")
        ax.set_ylim(-0.15, 1.05)
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="lower right")

    fig.suptitle("Clustering: ARI vs. Components (Controlled Simulation, Varying SNR)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    savepath = os.path.join(FIGURES_DIR, "clustering_ari.png")
    fig.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {savepath}")


if __name__ == "__main__":
    main()

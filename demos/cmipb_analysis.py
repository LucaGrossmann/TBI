"""
CMI-PB analysis — run TBI on the CMI-PB multi-omics vaccination dataset.

Usage
-----
    python demos/cmipb_analysis.py
    python demos/cmipb_analysis.py --days 0 1 3 14
    python demos/cmipb_analysis.py --energy 0.90
"""

import os
import time
import numpy as np

from TBI import TBI_I, TBI_II, matrix_MCIA
from TBI.data.cmipb import build_cmipb_tensor
from TBI.normalization import default_normalize, no_normalize
from TBI.star_M import mode3
from TBI.analysis_utils import (
    dct_matrix, scree_plot, score_scatter_grid, top_loadings_table,
    block_contribution_bar, block_variance_contributions, efficiency_plot,
)

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")


# ---------------------------------------------------------------------------
# Track 1: CMI-PB
# ---------------------------------------------------------------------------

def run_cmipb_analysis(days=None, energy=0.95, max_iter=15):
    """
    Fetch CMI-PB data, build block tensor, run TBI-I and TBI-II.
    """
    if days is None:
        days = [0, 1, 3, 14]

    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("=" * 60)
    print("Track 1: CMI-PB Multi-Omics Analysis")
    print("=" * 60)

    # Use 4 assays with good coverage:
    # ab_titer, cell_freq, cytokine_olink, cytokine_legendplex
    # Gene expression is too large to fetch via API;
    # t_cell_polarization/activation have poor timepoint coverage
    assays = ["ab_titer", "cell_freq", "cytokine_olink", "cytokine_legendplex"]

    # Build tensor
    X, b, meta = build_cmipb_tensor(days=days, assays=assays)
    m, p, n = X.shape
    assay_names = meta["assay_names"]
    variable_names = meta["variable_names"]
    block_sizes = meta["block_sizes"]
    subject_ids = meta["subject_ids"]

    print(f"\nTensor: ({m}, {p}, {n})")
    print(f"Blocks: {list(zip(assay_names, block_sizes))}")

    # DCT matrix
    M = dct_matrix(n)

    # Show block variance BEFORE and AFTER normalization
    X_hat = mode3(X, M)
    fracs_before = block_variance_contributions(X_hat, b)

    print("\nBlock variance fractions BEFORE normalization:")
    for name, frac in zip(assay_names, fracs_before):
        print(f"  {name}: {frac:.4f} ({frac*100:.1f}%)")

    X_hat_norm = default_normalize(X_hat, b)
    fracs_after = block_variance_contributions(X_hat_norm, b)

    print("\nBlock variance fractions AFTER normalization:")
    for name, frac in zip(assay_names, fracs_after):
        print(f"  {name}: {frac:.4f} ({frac*100:.1f}%)")

    block_contribution_bar(
        fracs_before, assay_names,
        title="CMI-PB: Block Variance BEFORE Normalization",
        savepath=os.path.join(FIGURES_DIR, "cmipb_blocks_before.png"),
    )
    block_contribution_bar(
        fracs_after, assay_names,
        title="CMI-PB: Block Variance AFTER Normalization",
        savepath=os.path.join(FIGURES_DIR, "cmipb_blocks_after.png"),
    )

    # --- TBI-I ---
    print(f"\n--- Running TBI-I (energy={energy}, max_iter={max_iter}) ---")
    start = time.perf_counter()
    result_I = TBI_I(X, b, M, energy=energy, max_iter=max_iter)
    elapsed_I = time.perf_counter() - start
    print(f"  Iterations: {result_I.n_iter}, Time: {elapsed_I:.3f}s")
    cum_I = result_I.variance_explained.sum() / result_I.total_variance
    print(f"  Cumulative variance: {cum_I:.4f} ({cum_I*100:.1f}%)")

    # --- TBI-II ---
    print(f"\n--- Running TBI-II (energy={energy}, max_iter={max_iter*n}) ---")
    start = time.perf_counter()
    result_II = TBI_II(X, b, M, energy=energy, max_iter=max_iter * n)
    elapsed_II = time.perf_counter() - start
    print(f"  Iterations: {result_II.n_iter}, Time: {elapsed_II:.3f}s")
    cum_II = result_II.variance_explained.sum() / result_II.total_variance
    print(f"  Cumulative variance: {cum_II:.4f} ({cum_II*100:.1f}%)")
    print(f"  Sheet selections: {result_II.sheet_indices}")

    # --- Matrix MCIA baseline ---
    print(f"\n--- Running Matrix MCIA (energy={energy}, max_iter={max_iter}) ---")
    start = time.perf_counter()
    result_mcia = matrix_MCIA(X, b, energy=energy, max_iter=max_iter)
    elapsed_mcia = time.perf_counter() - start
    cum_mcia = result_mcia.variance_explained.sum() / result_mcia.total_variance
    print(f"  Iterations: {result_mcia.n_iter}, Time: {elapsed_mcia:.3f}s")
    print(f"  Cumulative variance: {cum_mcia:.4f} ({cum_mcia*100:.1f}%)")

    # --- Scree plot ---
    scree_plot(
        {
            "TBI-I": result_I.variance_explained,
            "TBI-II": result_II.variance_explained,
            "Matrix MCIA": result_mcia.variance_explained,
        },
        total_variances={
            "TBI-I": result_I.total_variance,
            "TBI-II": result_II.total_variance,
            "Matrix MCIA": result_mcia.total_variance,
        },
        title="CMI-PB: TBI vs Matrix MCIA",
        savepath=os.path.join(FIGURES_DIR, "cmipb_scree.png"),
    )

    # --- Efficiency plot ---
    efficiency_plot(
        {
            "TBI-I": result_I.variance_explained,
            "TBI-II": result_II.variance_explained,
            "Matrix MCIA": result_mcia.variance_explained,
        },
        total_variances={
            "TBI-I": result_I.total_variance,
            "TBI-II": result_II.total_variance,
            "Matrix MCIA": result_mcia.total_variance,
        },
        storage_per_component={
            "TBI-I": (m + p) * n, "TBI-II": m + p,
            "Matrix MCIA": m + p * n,
        },
        title="CMI-PB: Variance Explained vs Storage (Scores + Loadings)",
        savepath=os.path.join(FIGURES_DIR, "cmipb_efficiency.png"),
    )

    # --- Score grid (all methods, first two components) ---
    # Color by infancy vaccine type if available
    subject_meta = meta.get("subject_meta", {})
    vac_map = {"wP": 0, "aP": 1}
    vac_names = {0: "wP", 1: "aP"}
    labels = np.array([vac_map.get(subject_meta.get(sid, {}).get("infancy_vac"), -1)
                       for sid in subject_ids])

    scores_dict = {}
    if result_I.n_iter >= 2:
        scores_dict["TBI-I"] = result_I.global_scores[:, :2, 0]
    if result_II.n_iter >= 2:
        scores_dict["TBI-II"] = result_II.global_scores[:, :2]
    if result_mcia.n_iter >= 2:
        scores_dict["Matrix MCIA"] = result_mcia.scores[:, :2]

    if scores_dict:
        plot_labels = labels if np.any(labels >= 0) else None
        valid = labels >= 0 if plot_labels is not None else None
        if valid is not None:
            filtered_scores = {k: v[valid] for k, v in scores_dict.items()}
            score_scatter_grid(
                filtered_scores, labels[valid], vac_names,
                title="CMI-PB: Score Comparison by Vaccine Type",
                savepath=os.path.join(FIGURES_DIR, "cmipb_scores_grid.png"),
            )
        else:
            score_scatter_grid(
                scores_dict,
                title="CMI-PB: Score Comparison",
                savepath=os.path.join(FIGURES_DIR, "cmipb_scores_grid.png"),
            )

    # --- Top loadings ---
    print("\n--- Top loadings (Component 1) ---")
    top = top_loadings_table(result_I.global_loadings, variable_names, k=15, component=0)
    for var_name, val in top:
        print(f"  {val:+.4f}  {var_name}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("CMI-PB SUMMARY")
    print("=" * 60)
    print(f"{'Method':<12} {'Iters':>6} {'Time':>8} {'Cum Var':>10}")
    print("-" * 38)
    print(f"{'TBI-I':<12} {result_I.n_iter:>6d} {elapsed_I:>7.3f}s {cum_I*100:>8.1f}%")
    print(f"{'TBI-II':<12} {result_II.n_iter:>6d} {elapsed_II:>7.3f}s {cum_II*100:>8.1f}%")
    print(f"{'Matrix MCIA':<12} {result_mcia.n_iter:>6d} {elapsed_mcia:>7.3f}s {cum_mcia*100:>8.1f}%")

    return result_I, result_II


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TBI Biological Data Demo")
    parser.add_argument("--days", nargs="+", type=int, default=None,
                        help="CMI-PB timepoints (default: 0 1 3 14)")
    parser.add_argument("--energy", type=float, default=0.95)
    args = parser.parse_args()

    run_cmipb_analysis(days=args.days, energy=args.energy)

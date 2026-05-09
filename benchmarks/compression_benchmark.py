"""
Compression benchmark: relative Frobenius error vs. compression ratio.

For each (method, dataset) pair, extracts components 1..max_iter and plots
how reconstruction quality improves as more storage is used. This directly
tests Kilmer et al. (2021, PNAS 118(28)) Theorem 5.3: tensor t-SVDM achieves
at least as good approximation as matrix SVD for the same storage budget.

Metrics
-------
- Relative error: epsilon_j = sqrt(1 - sum(sigma_i^2, i=1..j) / V_total)
  Standard in tensor decomposition literature (Kolda & Bader 2009, SIAM Review
  51(3):455-500). The Eckart-Young optimality proof uses this metric.

- Compression ratio: r_j = j * s / (m * p * n), where s = storage per component.
  Normalizes storage by the original tensor size, enabling fair comparison
  between methods with different storage-per-component costs (e.g., TBI-I
  stores (m+p)*n per component while TBI-II stores only m+p).

Output
------
Single figure (figures/benchmarks/compression_curves.png) with one subplot
per dataset. Each subplot shows one curve per method. Lower-left is better.

Usage
-----
    conda run -n claude python benchmarks/compression_benchmark.py
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from TBI import TBI_I, TBI_II, matrix_MCIA, load_dataset
from TBI.analysis_utils import dct_matrix, storage_per_component
from TBI.metrics import reconstruction_error_from_variance
from TBI.result_types import adapt_tbi_i, adapt_tbi_ii, adapt_mcia, BenchmarkResult

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures", "benchmarks")
ENERGY = 0.99
MAX_ITER = 15
DATASETS = ["cmipb", "suez2018"]


# ---------------------------------------------------------------------------
# Method runners
# ---------------------------------------------------------------------------

def run_tbi_i(X, b, M):
    t0 = time.perf_counter()
    result = TBI_I(X, b, M, energy=ENERGY, max_iter=MAX_ITER)
    elapsed = time.perf_counter() - t0
    return adapt_tbi_i(result, elapsed)


def run_tbi_ii(X, b, M):
    n = X.shape[2]
    t0 = time.perf_counter()
    result = TBI_II(X, b, M, energy=ENERGY, max_iter=MAX_ITER * n)
    elapsed = time.perf_counter() - t0
    return adapt_tbi_ii(result, elapsed)


def run_matrix_mcia(X, b, M):
    t0 = time.perf_counter()
    result = matrix_MCIA(X, b, energy=ENERGY, max_iter=MAX_ITER)
    elapsed = time.perf_counter() - t0
    return adapt_mcia(result, elapsed)


def run_block_tpls(X, b, M):
    try:
        from TBI.baselines.block_tpls import block_tpls
    except ImportError:
        return None
    t0 = time.perf_counter()
    result = block_tpls(X, b, M=M, n_components=MAX_ITER)
    elapsed = time.perf_counter() - t0
    return BenchmarkResult(
        method_name="block-tPLS",
        scores=result.scores,
        variance_explained=result.variance_explained,
        total_variance=result.total_variance,
        n_iter=result.n_iter,
        elapsed_seconds=elapsed,
    )


METHODS = [
    ("TBI-I", run_tbi_i, True),
    ("TBI-II", run_tbi_ii, True),
    ("Matrix MCIA", run_matrix_mcia, False),
    ("block-tPLS", run_block_tpls, True),
]


# ---------------------------------------------------------------------------
# Compression curves
# ---------------------------------------------------------------------------

def compute_compression_curve(result, m, p, n):
    """Compute (compression_ratio, relative_error) arrays for a BenchmarkResult."""
    var_exp = result.variance_explained
    total_var = result.total_variance
    spc = storage_per_component(result.method_name, m, p, n)
    total_elements = m * p * n

    n_comp = len(var_exp)
    ratios = np.zeros(n_comp)
    errors = np.zeros(n_comp)

    for j in range(n_comp):
        ratios[j] = (j + 1) * spc / total_elements
        errors[j] = reconstruction_error_from_variance(var_exp[:j + 1], total_var)

    return ratios, errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, len(DATASETS), figsize=(5 * len(DATASETS), 4.5),
                             squeeze=False)
    axes = axes[0]

    colors = {"TBI-I": "#1f77b4", "TBI-II": "#ff7f0e",
              "Matrix MCIA": "#2ca02c", "block-tPLS": "#d62728"}

    for col, ds_name in enumerate(DATASETS):
        ax = axes[col]
        print(f"\n{'='*50}")
        print(f"Dataset: {ds_name}")

        try:
            X, b, meta = load_dataset(ds_name)
        except Exception as e:
            print(f"  [ERROR] Could not load {ds_name}: {e}")
            ax.set_title(f"{ds_name} (failed)")
            continue

        m, p, n = X.shape
        M = dct_matrix(n)
        print(f"  Shape: ({m}, {p}, {n})")

        for method_name, runner, requires_tensor in METHODS:
            if requires_tensor and n < 2:
                continue
            try:
                result = runner(X, b, M)
                if result is None:
                    continue
                ratios, errors = compute_compression_curve(result, m, p, n)
                ax.plot(ratios, errors, "o-", label=method_name,
                        color=colors.get(method_name), markersize=3, linewidth=1.5)
                print(f"  {method_name:<15s}  components={result.n_iter:3d}  "
                      f"final_error={errors[-1]:.4f}  "
                      f"final_ratio={ratios[-1]:.4f}")
            except Exception as e:
                print(f"  [error] {method_name}: {e}")

        ax.set_xlabel("Compression Ratio")
        if col == 0:
            ax.set_ylabel("Relative Error")
        ax.set_title(ds_name)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")

    fig.suptitle("Compression: Relative Error vs. Compression Ratio",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    savepath = os.path.join(FIGURES_DIR, "compression_curves.png")
    fig.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {savepath}")


if __name__ == "__main__":
    main()

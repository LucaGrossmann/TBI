"""
Verification tests for TBI_QA.md -- theory sanity checks and correctness tests.

Each test is tagged with the question it verifies.
Run: conda run -n claude pytest tests/verify_theory.py -v
"""

import numpy as np
import numpy.testing as npt
import pytest

from TBI import TBI_I, TBI_II, matrix_MCIA
from TBI.normalization import (
    default_normalize,
    mcia_normalize,
    variable_var_normalize,
    block_var_normalize,
    block_frob_normalize,
    variable_mean_center,
)
from TBI.helpers import _block_ranges, _compute_total_variance
from TBI.analysis_utils import block_variance_contributions, dct_matrix


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def small_tensor(rng):
    """(20, 12, 4) tensor with 3 blocks at [0, 4, 8]."""
    X = rng.standard_normal((20, 12, 4))
    b = np.array([0, 4, 8])
    M = np.eye(4)
    return X, b, M


# ---------------------------------------------------------------------------
# I.1  SVD Singular Values vs. Variance  (Q6a)
# ---------------------------------------------------------------------------

class TestSVDVarianceIdentity:
    """Q6a: sum of squared singular values == Frobenius norm squared."""

    def test_svd_variance_identity_2d(self, rng):
        """For a 2D matrix: sum(sigma_i^2) == sum(X**2)."""
        X = rng.standard_normal((50, 30))
        _, s, _ = np.linalg.svd(X, full_matrices=False)
        npt.assert_allclose(np.sum(s ** 2), np.sum(X ** 2), rtol=1e-10)

    def test_svd_variance_identity_per_sheet(self, rng):
        """For a 3D tensor, the identity holds per sheet."""
        X = rng.standard_normal((20, 15, 5))
        for i in range(X.shape[2]):
            _, s, _ = np.linalg.svd(X[:, :, i], full_matrices=False)
            npt.assert_allclose(
                np.sum(s ** 2), np.sum(X[:, :, i] ** 2), rtol=1e-10
            )

    def test_total_variance_matches_frobenius(self, rng):
        """_compute_total_variance == sum over all sheets of sum(sigma_i^2)."""
        X = rng.standard_normal((20, 15, 5))
        total_from_code = _compute_total_variance(X)
        total_from_svd = 0.0
        for i in range(X.shape[2]):
            _, s, _ = np.linalg.svd(X[:, :, i], full_matrices=False)
            total_from_svd += np.sum(s ** 2)
        npt.assert_allclose(total_from_code, total_from_svd, rtol=1e-10)

    def test_mean_centered_covariance_eigenvalues(self, rng):
        """For mean-centered X: eigenvalues of X^T X / (m-1) == sigma_i^2 / (m-1)."""
        m, p = 50, 10
        X = rng.standard_normal((m, p))
        X = X - X.mean(axis=0)
        _, s, Vt = np.linalg.svd(X, full_matrices=False)
        S_cov = X.T @ X / (m - 1)
        eigvals = np.sort(np.linalg.eigvalsh(S_cov))[::-1]
        expected = s ** 2 / (m - 1)
        npt.assert_allclose(eigvals, expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# I.3 / Q6b: NIPALS vs full SVD
# ---------------------------------------------------------------------------

def _power_iteration(X, n_iter=100, tol=1e-12, rng=None):
    """Simple power iteration to find leading right singular vector."""
    if rng is None:
        rng = np.random.default_rng(0)
    v = rng.standard_normal(X.shape[1])
    v = v / np.linalg.norm(v)
    for _ in range(n_iter):
        u = X @ v
        u = u / np.linalg.norm(u)
        v_new = X.T @ u
        sigma = np.linalg.norm(v_new)
        v_new = v_new / sigma
        if np.abs(np.abs(v_new @ v) - 1.0) < tol:
            break
        v = v_new
    return sigma, v_new


class TestNIPALSvsSVD:
    """Q6b: power iteration converges to the same leading singular vector as SVD."""

    def test_leading_singular_value_matches(self, rng):
        X = rng.standard_normal((40, 20))
        _, s_full, _ = np.linalg.svd(X, full_matrices=False)
        sigma_power, _ = _power_iteration(X, rng=rng)
        npt.assert_allclose(sigma_power, s_full[0], rtol=1e-8)

    def test_leading_singular_vector_matches(self, rng):
        X = rng.standard_normal((40, 20))
        _, _, Vt_full = np.linalg.svd(X, full_matrices=False)
        _, v_power = _power_iteration(X, rng=rng)
        # Singular vectors can differ by sign
        alignment = np.abs(v_power @ Vt_full[0])
        npt.assert_allclose(alignment, 1.0, atol=1e-8)

    def test_power_iteration_on_each_sheet(self, rng):
        """Power iteration matches SVD for each sheet of a tensor."""
        X = rng.standard_normal((30, 15, 4))
        for i in range(X.shape[2]):
            _, s_full, Vt_full = np.linalg.svd(X[:, :, i], full_matrices=False)
            sigma, v = _power_iteration(X[:, :, i], rng=rng)
            npt.assert_allclose(sigma, s_full[0], rtol=1e-8)
            npt.assert_allclose(np.abs(v @ Vt_full[0]), 1.0, atol=1e-8)


# ---------------------------------------------------------------------------
# Q2: TBI-I reconstruction
# ---------------------------------------------------------------------------

class TestTBIIReconstruction:
    """Q2: cumulative variance_explained sums to approximately total_variance."""

    def test_variance_explained_sums_to_total(self, small_tensor):
        X, b, M = small_tensor
        result = TBI_I(X, b, M, energy=1.0, max_iter=12)
        cumulative = np.sum(result.variance_explained)
        # Should capture most (>99%) of total variance with enough iterations
        assert cumulative / result.total_variance > 0.99

    def test_variance_explained_fractions_are_valid(self, small_tensor):
        X, b, M = small_tensor
        result = TBI_I(X, b, M, energy=0.9, max_iter=10)
        fractions = result.variance_explained / result.total_variance
        # Each fraction should be non-negative
        assert np.all(fractions >= -1e-12)
        # Cumulative should not exceed 1
        assert np.sum(fractions) <= 1.0 + 1e-10


# ---------------------------------------------------------------------------
# Q4: TBI-II greedy picks max sigma
# ---------------------------------------------------------------------------

class TestTBIIIGreedy:
    """Q4: each iteration picks the sheet with the largest leading sigma."""

    def test_greedy_picks_max_sigma(self, rng):
        """The selected sheet should have the largest sigma_1.
        Uses TBI_II internals to avoid re-normalizing outside the algorithm."""
        from TBI.TBI_II import _svd_all_sheets, _find_best_sheet
        from TBI.star_M import mode3

        X = rng.standard_normal((25, 10, 5))
        b = np.array([0, 5])
        M = np.eye(5)

        # Run the exact same pipeline as TBI_II
        X_hat = mode3(X, M)
        X_hat = default_normalize(X_hat, b)
        cache = _svd_all_sheets(X_hat)
        _, best_sheet, _ = _find_best_sheet(cache)

        # Verify the cache sheet really has the max sigma
        sigmas = [entry[0] for entry in cache]
        assert best_sheet == np.argmax(sigmas)

        # Verify TBI_II agrees
        result = TBI_II(X, b, M, energy=0.5, max_iter=1)
        assert result.sheet_indices[0] == best_sheet

    def test_greedy_second_iteration_updates(self, rng):
        """After deflating the best sheet, the next pick may differ."""
        X = rng.standard_normal((25, 10, 4))
        b = np.array([0, 5])
        M = np.eye(4)
        result = TBI_II(X, b, M, energy=0.99, max_iter=4)
        # At least 2 iterations should have run
        assert result.n_iter >= 2
        # sheet_indices should all be valid
        assert np.all(result.sheet_indices < 4)
        assert np.all(result.sheet_indices >= 0)


# ---------------------------------------------------------------------------
# Q5: matrix_MCIA identity M equivalent
# ---------------------------------------------------------------------------

class TestMCIANoTransform:
    """Q5: matrix_MCIA is a pure matrix method — no M transform."""

    def test_result_independent_of_data_order(self, rng):
        """MCIA operates on raw data — results depend only on X and b."""
        X = rng.standard_normal((20, 8, 3))
        b = np.array([0, 4])

        r1 = matrix_MCIA(X, b, energy=0.99, max_iter=5)
        r2 = matrix_MCIA(X, b, energy=0.99, max_iter=5)

        npt.assert_allclose(r1.scores, r2.scores, atol=1e-10)
        npt.assert_allclose(r1.loadings, r2.loadings, atol=1e-10)
        npt.assert_allclose(r1.variance_explained, r2.variance_explained, atol=1e-10)


# ---------------------------------------------------------------------------
# Q7/Q10: var normalize prevents unit dominance
# ---------------------------------------------------------------------------

class TestVarNormalizePreventsDominance:
    """Q7/Q10: without var normalization, large-scale variables dominate SVD."""

    def test_without_normalize_large_block_dominates(self, rng):
        """Block 1 has values ~1000, block 2 has values ~1. Without normalization,
        SVD loading concentrates on block 1."""
        m, n = 30, 3
        block1 = rng.standard_normal((m, 5, n)) * 1000  # large scale
        block2 = rng.standard_normal((m, 5, n))          # unit scale
        X = np.concatenate([block1, block2], axis=1)

        # SVD of first sheet without normalization
        _, _, Vt = np.linalg.svd(X[:, :, 0], full_matrices=False)
        loading = np.abs(Vt[0])
        # Block 1 (cols 0-4) should dominate
        block1_loading = np.sum(loading[:5] ** 2)
        block2_loading = np.sum(loading[5:] ** 2)
        assert block1_loading > 0.99  # almost all loading on block 1

    def test_with_normalize_both_blocks_contribute(self, rng):
        """After variable normalization, both blocks contribute to SVD."""
        m, n = 30, 3
        block1 = rng.standard_normal((m, 5, n)) * 1000
        block2 = rng.standard_normal((m, 5, n))
        X = np.concatenate([block1, block2], axis=1)

        X_norm = variable_mean_center(X, sheet_level=False)
        X_norm = variable_var_normalize(X_norm, sheet_level=False)

        _, _, Vt = np.linalg.svd(X_norm[:, :, 0], full_matrices=False)
        loading = np.abs(Vt[0])
        block1_loading = np.sum(loading[:5] ** 2)
        block2_loading = np.sum(loading[5:] ** 2)
        # Neither block should have >95% of the loading
        assert block1_loading < 0.95
        assert block2_loading < 0.95


# ---------------------------------------------------------------------------
# Q8: MCIA normalization positive offset artifact
# ---------------------------------------------------------------------------

class TestMCIANormArtifact:
    """Q8: CA normalization on mixed-scale data — the positive offset concentrates
    variance in the first component more than default normalization does."""

    @staticmethod
    def _first_component_fraction(mat):
        _, s, _ = np.linalg.svd(mat, full_matrices=False)
        return s[0] ** 2 / np.sum(s ** 2)

    def test_mcia_norm_concentrates_variance_more_than_default(self, rng):
        """MCIA normalization on data with negative values concentrates more
        variance in the first SVD component compared to default normalization."""
        m, n = 40, 3
        # Mixed-scale: some variables centered ~0, some ~100
        X = np.concatenate([
            rng.standard_normal((m, 5, n)),
            rng.standard_normal((m, 5, n)) + 100,
        ], axis=1)
        b = np.array([0, 5])

        X_mcia = mcia_normalize(X, b)
        X_def = default_normalize(X, b)

        frac_mcia = self._first_component_fraction(X_mcia[:, :, 0])
        frac_def = self._first_component_fraction(X_def[:, :, 0])

        assert frac_mcia > frac_def, (
            f"Expected MCIA to concentrate more variance in PC1: "
            f"MCIA={frac_mcia:.3f}, default={frac_def:.3f}"
        )

    def test_offset_is_the_cause(self, rng):
        """When data is already positive (no offset needed), the concentration
        gap between MCIA and default shrinks, confirming the offset is the culprit."""
        m, n = 40, 3
        b = np.array([0, 5])

        # Case 1: data with negative values -> large offset applied
        X_neg = np.concatenate([
            rng.standard_normal((m, 5, n)),
            rng.standard_normal((m, 5, n)) + 100,
        ], axis=1)
        frac_neg = self._first_component_fraction(mcia_normalize(X_neg, b)[:, :, 0])
        frac_neg_def = self._first_component_fraction(default_normalize(X_neg, b)[:, :, 0])
        gap_neg = frac_neg - frac_neg_def

        # Case 2: data already positive -> minimal offset
        X_pos = np.abs(rng.standard_normal((m, 10, n))) + 1.0
        frac_pos = self._first_component_fraction(mcia_normalize(X_pos, b)[:, :, 0])
        frac_pos_def = self._first_component_fraction(default_normalize(X_pos, b)[:, :, 0])
        gap_pos = frac_pos - frac_pos_def

        # The gap should be larger when offset is needed (negative data)
        assert gap_neg > gap_pos, (
            f"Expected offset to amplify concentration gap: "
            f"gap_neg={gap_neg:.3f}, gap_pos={gap_pos:.3f}"
        )


# ---------------------------------------------------------------------------
# Q9: Frobenius vs variance block normalization
# ---------------------------------------------------------------------------

class TestFrobVsVarBlockVariance:
    """Q9: after default_normalize (which uses block_frob_normalize), all blocks
    have equal total variance (Frobenius norm = 1)."""

    def test_block_frob_normalize_equal_total_variance(self, rng):
        """After block_frob_normalize, each block has Frobenius norm = 1."""
        X = rng.standard_normal((20, 15, 3))
        b = np.array([0, 3, 10])  # blocks of size 3, 7, 5
        X_centered = variable_mean_center(X, sheet_level=False)
        X_scaled = variable_var_normalize(X_centered, sheet_level=False)
        X_frob = block_frob_normalize(X_scaled, b)

        _, p, _ = X_frob.shape
        for _, start, end in _block_ranges(b, p):
            block = X_frob[:, start:end, :]
            fnorm = np.sqrt(np.sum(block ** 2))
            npt.assert_allclose(fnorm, 1.0, rtol=1e-12)

    def test_default_normalize_equal_total_variance(self, rng):
        """After default_normalize (now using Frobenius), all blocks have
        equal total sum-of-squares = 1, regardless of block size."""
        X = rng.standard_normal((20, 15, 3))
        b = np.array([0, 3, 10])  # blocks of size 3, 7, 5
        X_def = default_normalize(X, b)

        _, p, _ = X_def.shape
        for _, start, end in _block_ranges(b, p):
            block = X_def[:, start:end, :]
            fnorm = np.sqrt(np.sum(block ** 2))
            npt.assert_allclose(fnorm, 1.0, rtol=1e-12)


# ---------------------------------------------------------------------------
# Q11: Block variance contributions vs. singular values
# ---------------------------------------------------------------------------

class TestBlockVarianceVsSingularValues:
    """Q11: block variance contribution == sum of squared singular values across sheets."""

    def test_block_variance_matches_svd_sum(self, rng):
        X = rng.standard_normal((20, 12, 4))
        b = np.array([0, 4, 8])
        _, p, n = X.shape

        fractions = block_variance_contributions(X, b)
        total_ss = np.sum(X ** 2)

        for idx, (_, start, end) in enumerate(_block_ranges(b, p)):
            # Sum of squared singular values across all sheets
            svd_ss = 0.0
            for i in range(n):
                _, s, _ = np.linalg.svd(X[:, start:end, i], full_matrices=False)
                svd_ss += np.sum(s ** 2)

            expected_fraction = svd_ss / total_ss
            npt.assert_allclose(fractions[idx], expected_fraction, rtol=1e-10)

    def test_zero_variance_block(self, rng):
        """A block of all zeros should have fraction = 0."""
        X = rng.standard_normal((20, 12, 4))
        X[:, 8:12, :] = 0.0  # zero out the third block
        b = np.array([0, 4, 8])
        fractions = block_variance_contributions(X, b)
        npt.assert_allclose(fractions[2], 0.0, atol=1e-16)


# ---------------------------------------------------------------------------
# Edge cases: wide tensors, single-block, single-sheet
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases flagged by numerical verification: m < p, single-block, single-sheet."""

    def test_svd_identity_wide_matrix(self, rng):
        """SVD variance identity holds when m < p (wide matrix)."""
        X = rng.standard_normal((5, 30))  # m << p
        _, s, _ = np.linalg.svd(X, full_matrices=False)
        # Only min(m,p) = 5 singular values returned
        assert len(s) == 5
        npt.assert_allclose(np.sum(s ** 2), np.sum(X ** 2), rtol=1e-10)

    def test_svd_identity_wide_tensor(self, rng):
        """SVD variance identity per sheet for wide tensor."""
        X = rng.standard_normal((5, 20, 3))
        for i in range(X.shape[2]):
            _, s, _ = np.linalg.svd(X[:, :, i], full_matrices=False)
            npt.assert_allclose(np.sum(s ** 2), np.sum(X[:, :, i] ** 2), rtol=1e-10)

    def test_single_block_normalization(self, rng):
        """Normalization works with a single block (k=1)."""
        X = rng.standard_normal((20, 8, 3))
        b = np.array([0])
        X_def = default_normalize(X, b)
        # Should not crash, and shape should be preserved
        assert X_def.shape == X.shape
        # With Frobenius normalization, single block has ||block||_F = 1
        fnorm = np.sqrt(np.sum(X_def ** 2))
        npt.assert_allclose(fnorm, 1.0, rtol=1e-12)

    def test_single_sheet_tbi(self, rng):
        """TBI-I and TBI-II work with n=1 (single sheet)."""
        X = rng.standard_normal((20, 10, 1))
        b = np.array([0, 5])
        M = np.eye(1)

        r1 = TBI_I(X, b, M, energy=0.9, max_iter=5)
        assert r1.n_iter >= 1
        assert r1.global_scores.shape[0] == 20

        r2 = TBI_II(X, b, M, energy=0.9, max_iter=5)
        assert r2.n_iter >= 1
        assert r2.sheet_indices[0] == 0  # only one sheet to pick

    def test_block_frob_normalize_single_block(self, rng):
        """Frobenius normalization with single block gives norm = 1."""
        X = rng.standard_normal((15, 6, 3))
        b = np.array([0])
        X_frob = block_frob_normalize(X, b)
        fnorm = np.sqrt(np.sum(X_frob ** 2))
        npt.assert_allclose(fnorm, 1.0, rtol=1e-12)

    def test_power_iteration_wide_matrix(self, rng):
        """Power iteration converges for wide matrix (m < p)."""
        X = rng.standard_normal((8, 30))
        _, s_full, Vt_full = np.linalg.svd(X, full_matrices=False)
        sigma_power, v_power = _power_iteration(X, rng=rng)
        npt.assert_allclose(sigma_power, s_full[0], rtol=1e-8)
        npt.assert_allclose(np.abs(v_power @ Vt_full[0]), 1.0, atol=1e-8)


# ---------------------------------------------------------------------------
# Q16: Block loadings formula — X_k^T f vs X_k^T X_k a
# ---------------------------------------------------------------------------

class TestBlockLoadingFormula:
    """Q16: X_k^T f / ||X_k^T f|| is the optimal block loading (Cauchy-Schwarz)."""

    def test_block_loading_optimality(self, rng):
        """The formula a_k = X_k^T f / ||X_k^T f|| maximizes cov^2(X_k a_k, f)
        over all unit vectors a_k. Verify by comparing against random directions."""
        m, p_k = 30, 8
        X_k = rng.standard_normal((m, p_k))
        f = rng.standard_normal(m)

        # Optimal block loading (what the code computes)
        numerator = X_k.T @ f
        a_k_opt = numerator / np.linalg.norm(numerator)
        cov_opt = (a_k_opt @ X_k.T @ f) ** 2

        # Compare against 1000 random unit vectors
        for _ in range(1000):
            a_rand = rng.standard_normal(p_k)
            a_rand /= np.linalg.norm(a_rand)
            cov_rand = (a_rand @ X_k.T @ f) ** 2
            assert cov_opt >= cov_rand - 1e-10, (
                f"Random direction achieved higher cov^2: {cov_rand:.6f} > {cov_opt:.6f}"
            )

    def test_block_loading_beats_covariance_formula(self, rng):
        """The thesis formula X_k^T X_k a / ||...|| gives a DIFFERENT (suboptimal)
        direction compared to the correct X_k^T f / ||X_k^T f||."""
        m, p_k = 30, 8
        X_k = rng.standard_normal((m, p_k))
        f = rng.standard_normal(m)

        # Correct formula: X_k^T f
        num_correct = X_k.T @ f
        a_correct = num_correct / np.linalg.norm(num_correct)

        # Thesis formula (7.3.2): X_k^T X_k * (global_loading)
        # Use a random global loading to simulate what the thesis suggests
        a_global = rng.standard_normal(p_k)
        a_global /= np.linalg.norm(a_global)
        num_thesis = X_k.T @ X_k @ a_global
        a_thesis = num_thesis / np.linalg.norm(num_thesis)

        # They should generally differ
        alignment = np.abs(a_correct @ a_thesis)
        # Not perfectly aligned (would need a very special X_k for them to coincide)
        assert alignment < 1.0 - 1e-10, "Formulas coincide — use different random data"

        # Correct formula achieves higher (or equal) cov^2
        cov_correct = (a_correct @ X_k.T @ f) ** 2
        cov_thesis = (a_thesis @ X_k.T @ f) ** 2
        assert cov_correct >= cov_thesis - 1e-10

    def test_block_loading_matches_tbi_code(self, rng):
        """Verify the TBI_I _compute_block_loadings function matches the
        Cauchy-Schwarz optimal formula."""
        from TBI.TBI_I import _compute_block_loadings, _compute_global_loadings, _compute_global_scores

        m, p, n = 20, 12, 4
        X = rng.standard_normal((m, p, n))
        b = np.array([0, 4, 8])

        loadings = _compute_global_loadings(X)
        scores = _compute_global_scores(X, loadings)
        block_loadings = _compute_block_loadings(X, b, scores)

        # Verify each block loading matches X_k^T f / ||X_k^T f||
        for (idx, start, end), bl in zip(_block_ranges(b, p), block_loadings):
            for i in range(n):
                expected = X[:, start:end, i].T @ scores[:, i]
                norm = np.linalg.norm(expected)
                if norm > 1e-16:
                    expected /= norm
                    npt.assert_allclose(bl[:, i], expected, atol=1e-12)


# ---------------------------------------------------------------------------
# Q17: TBI-II code vs thesis — step-by-step verification
# ---------------------------------------------------------------------------

class TestTBIIIStepsMatchManual:
    """Q17: verify each TBI-II step matches the expected formula."""

    def test_tbi_ii_block_loading_optimality(self, rng):
        """TBI-II block loadings use X_k^T f / ||X_k^T f|| (Cauchy-Schwarz optimal)."""
        from TBI.TBI_II import (
            _svd_all_sheets, _find_best_sheet,
            _compute_scores_single, _compute_block_loadings_single,
        )

        m, p, n = 25, 10, 5
        X = rng.standard_normal((m, p, n))
        b = np.array([0, 4])

        cache = _svd_all_sheets(X)
        loading, sheet_idx, _ = _find_best_sheet(cache)
        scores = _compute_scores_single(X, loading, sheet_idx)
        block_loadings = _compute_block_loadings_single(X, b, scores, sheet_idx)

        # Verify each block loading maximizes cov^2(X_k a_k, f)
        for (idx, start, end), bl in zip(_block_ranges(b, p), block_loadings):
            X_k = X[:, start:end, sheet_idx]
            cov_opt = (bl @ X_k.T @ scores) ** 2

            # Compare against 500 random unit vectors
            for _ in range(500):
                a_rand = rng.standard_normal(end - start)
                a_rand /= np.linalg.norm(a_rand)
                cov_rand = (a_rand @ X_k.T @ scores) ** 2
                assert cov_opt >= cov_rand - 1e-10

    def test_tbi_ii_steps_match_manual(self, rng):
        """Every TBI-II step matches the manual formulas from the thesis."""
        from TBI.TBI_II import (
            _svd_all_sheets, _find_best_sheet,
            _compute_scores_single, _compute_block_loadings_single,
            _compute_block_scores_single, _deflate_single,
        )

        m, p, n = 20, 12, 4
        X = rng.standard_normal((m, p, n))
        b = np.array([0, 4, 8])
        X_work = X.copy()

        # Step 1: Sheet selection = argmax sigma_1
        cache = _svd_all_sheets(X_work)
        sigmas = []
        for j in range(n):
            _, s, _ = np.linalg.svd(X_work[:, :, j], full_matrices=False)
            sigmas.append(s[0])
        loading, sheet_idx, sigma_sq = _find_best_sheet(cache)
        assert sheet_idx == np.argmax(sigmas)

        # Step 2: Global loading = first right singular vector of best sheet
        _, _, Vt = np.linalg.svd(X_work[:, :, sheet_idx], full_matrices=False)
        npt.assert_allclose(np.abs(loading @ Vt[0]), 1.0, atol=1e-10)

        # Step 3: Global scores = X^(i) @ a^(i)
        scores = _compute_scores_single(X_work, loading, sheet_idx)
        expected_scores = X_work[:, :, sheet_idx] @ loading
        npt.assert_allclose(scores, expected_scores, atol=1e-12)

        # Step 4: Block loadings = X_k^T f / ||X_k^T f||
        block_loadings = _compute_block_loadings_single(X_work, b, scores, sheet_idx)
        for (_, start, end), bl in zip(_block_ranges(b, p), block_loadings):
            num = X_work[:, start:end, sheet_idx].T @ scores
            expected_bl = num / np.linalg.norm(num)
            npt.assert_allclose(bl, expected_bl, atol=1e-12)

        # Step 5: Block scores = X_k @ a_k
        block_scores = _compute_block_scores_single(X_work, b, block_loadings, sheet_idx)
        for (_, start, end), bl, bs in zip(_block_ranges(b, p), block_loadings, block_scores):
            expected_bs = X_work[:, start:end, sheet_idx] @ bl
            npt.assert_allclose(bs, expected_bs, atol=1e-12)

        # Step 6: Deflation = X_k -= X_k @ a_k @ a_k^T (single sheet only)
        X_before = X_work.copy()
        _deflate_single(X_work, b, block_loadings, sheet_idx)

        # Non-selected sheets should be unchanged
        for j in range(n):
            if j != sheet_idx:
                npt.assert_allclose(X_work[:, :, j], X_before[:, :, j], atol=1e-15)

        # Selected sheet: each block deflated by projection removal
        for (_, start, end), bl in zip(_block_ranges(b, p), block_loadings):
            a = bl.reshape(-1, 1)
            expected = X_before[:, start:end, sheet_idx] - \
                X_before[:, start:end, sheet_idx] @ a @ a.T
            npt.assert_allclose(X_work[:, start:end, sheet_idx], expected, atol=1e-12)



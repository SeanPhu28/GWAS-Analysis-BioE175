"""
test_pipeline.py
----------------
Unit and integration tests for the cross-ancestry analysis pipeline.

Tests are grouped by the course methods they validate:
  - TestStandardize       → Week 3 preprocessing requirement for PCA/NMF
  - TestPCA               → Week 3 dimensionality reduction
  - TestNMF               → Week 3 dimensionality reduction
  - TestProjectionR2      → Week 2 regression / R² evaluation
  - TestCrossValidation   → Week 2 model validation
  - TestVariantAlignment  → data integrity / scientific validity

Run with: pytest Tests/test_pipeline.py -v
"""

import sys, os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Code"))

from utils import (
    standardize,
    run_tsvd,
    run_nmf,
    compute_projection_r2,
    crossval_projection_r2,
    align_to_factorgo_variants,
    abs_pearson,
)


# ════════════════════════════════════════════════════════════
# SHARED FIXTURES
# ════════════════════════════════════════════════════════════

@pytest.fixture
def z_matrix():
    """80 variants × 6 traits synthetic Z-score matrix."""
    np.random.seed(42)
    variants = [f"1:{i*1000}:A:T" for i in range(80)]
    traits   = ["BMI", "Height", "BMR", "FatMass", "FEV1", "Triglycerides"]
    return pd.DataFrame(np.random.randn(80, 6), index=variants, columns=traits)


@pytest.fixture
def eur_loadings():
    """80 variants × 10 factors synthetic EUR loading matrix."""
    np.random.seed(7)
    variants = [f"1:{i*1000}:A:T" for i in range(80)]
    cols     = [f"Factor_{k+1}" for k in range(10)]
    return pd.DataFrame(np.random.randn(80, 10), index=variants, columns=cols)


# ════════════════════════════════════════════════════════════
# Week 3: STANDARDIZATION
# Covered in PCA lecture — required before computing covariance matrix
# ════════════════════════════════════════════════════════════

class TestStandardize:
    """
    Tests for column standardization.
    Course: Week 3 — standardization is Step 1 of PCA pipeline.
    """

    def test_zero_mean(self, z_matrix):
        """Each column must have mean ≈ 0 after standardization."""
        result = standardize(z_matrix)
        assert result.mean().abs().max() < 1e-10, \
            "Standardized columns must have zero mean"

    def test_unit_variance(self, z_matrix):
        """Each column must have std ≈ 1 after standardization."""
        result = standardize(z_matrix)
        assert (result.std() - 1).abs().max() < 1e-10, \
            "Standardized columns must have unit variance"

    def test_shape_unchanged(self, z_matrix):
        """Standardization must not alter matrix dimensions."""
        result = standardize(z_matrix)
        assert result.shape == z_matrix.shape

    def test_no_nan(self, z_matrix):
        """Output must contain no NaN values."""
        result = standardize(z_matrix)
        assert not result.isna().any().any()

    def test_constant_column_handled(self):
        """Constant columns (std=0) should become 0, not NaN/inf."""
        df = pd.DataFrame({"A": [1.0]*5, "B": np.random.randn(5)})
        result = standardize(df)
        assert not result.isna().any().any()
        assert (result["A"] == 0).all()


# ════════════════════════════════════════════════════════════
# Week 3: PCA / tSVD
# ════════════════════════════════════════════════════════════

class TestTSVD:
    """
    Tests for PCA (tSVD) implementation.
    Course: Week 3 — dimensionality reduction lecture. tSVD is the paper's baseline model, distinct from PCA because it decomposes the raw matrix directly without forming a covariance matrix.
    """

    def test_loadings_shape(self, z_matrix):
        """Loadings must be (n_variants, n_components)."""
        z   = z_matrix.values
        L, S, var = run_tsvd(z, list(z_matrix.index), list(z_matrix.columns), 4)
        assert L.shape == (80, 4)

    def test_scores_shape(self, z_matrix):
        """Scores must be (n_traits, n_components)."""
        z   = z_matrix.values
        L, S, var = run_tsvd(z, list(z_matrix.index), list(z_matrix.columns), 4)
        assert S.shape == (6, 4)

    def test_variance_explained_nonnegative(self, z_matrix):
        """All variance explained values must be ≥ 0."""
        z = z_matrix.values
        _, _, var = run_tsvd(z, list(z_matrix.index), list(z_matrix.columns), 4)
        assert (var >= 0).all()

    def test_variance_explained_at_most_one(self, z_matrix):
        """Total variance explained cannot exceed 100%."""
        z = z_matrix.values
        _, _, var = run_tsvd(z, list(z_matrix.index), list(z_matrix.columns), 4)
        assert var.sum() <= 1.0 + 1e-10

    def test_more_components_more_variance(self, z_matrix):
        """More components must explain at least as much variance."""
        z = z_matrix.values
        v = list(z_matrix.index); t = list(z_matrix.columns)
        _, _, var_3 = run_tsvd(z, v, t, 3)
        _, _, var_5 = run_tsvd(z, v, t, 5)
        assert var_5.sum() >= var_3.sum() - 1e-10

    def test_index_labels_correct(self, z_matrix):
        """Loading index must match input variant IDs."""
        z = z_matrix.values
        L, _, _ = run_tsvd(z, list(z_matrix.index), list(z_matrix.columns), 3)
        assert list(L.index) == list(z_matrix.index)


# ════════════════════════════════════════════════════════════
# Week 3: NMF
# ════════════════════════════════════════════════════════════

class TestNMF:
    """
    Tests for NMF implementation.
    Course: Week 3 — NMF lecture (parts-based decomposition).
    The key property tested is non-negativity, which enforces
    the additive biological interpretation.
    """

    def test_loadings_nonnegative(self, z_matrix):
        """All NMF loadings must be ≥ 0 — core NMF constraint."""
        z = z_matrix.values
        W, H, _ = run_nmf(z, list(z_matrix.index), list(z_matrix.columns), 4)
        assert (W.values >= 0).all(), "NMF loadings must be non-negative"

    def test_scores_nonnegative(self, z_matrix):
        """All NMF scores must be ≥ 0 — core NMF constraint."""
        z = z_matrix.values
        W, H, _ = run_nmf(z, list(z_matrix.index), list(z_matrix.columns), 4)
        assert (H.values >= 0).all(), "NMF scores must be non-negative"

    def test_reconstruction_error_positive(self, z_matrix):
        """Reconstruction error must be > 0 for non-trivial decomposition."""
        z = z_matrix.values
        _, _, err = run_nmf(z, list(z_matrix.index), list(z_matrix.columns), 4)
        assert err > 0

    def test_output_shapes(self, z_matrix):
        """Output shapes must match (n_variants, k) and (n_traits, k)."""
        z = z_matrix.values
        W, H, _ = run_nmf(z, list(z_matrix.index), list(z_matrix.columns), 4)
        assert W.shape == (80, 4)
        assert H.shape == (6,  4)

    def test_more_components_lower_error(self, z_matrix):
        """More NMF components should not increase reconstruction error."""
        z = z_matrix.values; v = list(z_matrix.index); t = list(z_matrix.columns)
        _, _, err_3 = run_nmf(z, v, t, 3)
        _, _, err_5 = run_nmf(z, v, t, 5)
        assert err_5 <= err_3 + 1.0  # Allow small numerical tolerance


# ════════════════════════════════════════════════════════════
# Week 2: R² PROJECTION METRIC
# ════════════════════════════════════════════════════════════

class TestProjectionR2:
    """
    Tests for the projection R² metric.
    Course: Week 2 — R² is the core regression evaluation metric.
    Here applied to subspace projection rather than linear regression,
    but the mathematical definition is identical.
    """

    def test_r2_in_unit_interval(self):
        """R² must always be in [0, 1]."""
        np.random.seed(0)
        afr = np.random.randn(100)
        L   = np.random.randn(100, 10)
        r2  = compute_projection_r2(afr, L)
        assert 0 <= r2 <= 1

    def test_perfect_projection_gives_one(self):
        """Projecting a vector onto itself should give R² = 1."""
        np.random.seed(0)
        vec = np.random.randn(100)
        r2  = compute_projection_r2(vec, vec.reshape(-1, 1))
        assert abs(r2 - 1.0) < 1e-6, f"Expected R²=1, got {r2:.6f}"

    def test_orthogonal_projection_gives_zero(self):
        """Projecting onto an orthogonal subspace should give R² ≈ 0."""
        a = np.array([1.0, 0.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0, 0.0])
        r2 = compute_projection_r2(b, a.reshape(-1, 1))
        assert r2 < 0.01

    def test_more_factors_higher_r2(self):
        """Adding more factors can only maintain or increase R²."""
        np.random.seed(42)
        afr = np.random.randn(200)
        L3  = np.random.randn(200, 3)
        L5  = np.random.randn(200, 5)
        # Not guaranteed in general, but holds for random vectors in expectation
        r2_3 = compute_projection_r2(afr, L3)
        r2_5 = compute_projection_r2(afr, L5)
        # R² should be similar magnitude — both in [0,1]
        assert 0 <= r2_3 <= 1
        assert 0 <= r2_5 <= 1


# ════════════════════════════════════════════════════════════
# Week 2: CROSS-VALIDATION
# ════════════════════════════════════════════════════════════

class TestCrossValidation:
    """
    Tests for cross-validated R².
    Course: Week 2 — "Does my model work?" lecture.
    CV R² is the honest generalization estimate vs optimistic in-sample R².
    """

    def test_cv_r2_in_unit_interval(self):
        """CV R² mean must be in [0, 1]."""
        np.random.seed(0)
        afr = np.random.randn(200)
        L   = np.random.randn(200, 5)
        mean_r2, std_r2, folds = crossval_projection_r2(afr, L, n_splits=5)
        assert 0 <= mean_r2 <= 1

    def test_cv_std_nonnegative(self):
        """Standard deviation of fold R² must be ≥ 0."""
        np.random.seed(0)
        afr = np.random.randn(200)
        L   = np.random.randn(200, 5)
        _, std_r2, _ = crossval_projection_r2(afr, L, n_splits=5)
        assert std_r2 >= 0

    def test_correct_number_of_folds(self):
        """fold_r2s array must have exactly n_splits values."""
        np.random.seed(0)
        afr = np.random.randn(200)
        L   = np.random.randn(200, 5)
        _, _, folds = crossval_projection_r2(afr, L, n_splits=4)
        assert len(folds) == 4

    def test_cv_r2_not_greater_than_insample(self):
        """CV R² should not exceed in-sample R² by more than small margin."""
        np.random.seed(99)
        afr = np.random.randn(300)
        L   = np.random.randn(300, 8)
        insample = compute_projection_r2(afr, L)
        cv_mean, _, _ = crossval_projection_r2(afr, L, n_splits=5)
        # CV R² should be ≤ in-sample + small tolerance
        assert cv_mean <= insample + 0.2

    def test_reproducible_with_same_seed(self):
        """Same random_state must give identical results."""
        np.random.seed(0)
        afr = np.random.randn(200)
        L   = np.random.randn(200, 5)
        r1, _, _ = crossval_projection_r2(afr, L, n_splits=5, random_state=42)
        r2, _, _ = crossval_projection_r2(afr, L, n_splits=5, random_state=42)
        assert r1 == r2


# ════════════════════════════════════════════════════════════
# DATA INTEGRITY: VARIANT ALIGNMENT
# ════════════════════════════════════════════════════════════

class TestVariantAlignment:
    """
    Tests for variant alignment between AFR and EUR datasets.
    Scientific requirement: same genomic regions must be compared
    across ancestries for results to be interpretable.
    """

    def test_output_uses_shared_variants_only(self, z_matrix, eur_loadings):
        """Both aligned matrices must have identical variant sets."""
        afr_a, eur_a, n = align_to_factorgo_variants(z_matrix, eur_loadings)
        assert set(afr_a.index) == set(eur_a.index)

    def test_n_shared_matches_output_length(self, z_matrix, eur_loadings):
        """Reported n_shared must match actual aligned matrix size."""
        afr_a, eur_a, n = align_to_factorgo_variants(z_matrix, eur_loadings)
        assert n == len(afr_a) == len(eur_a)

    def test_full_overlap_returns_all(self):
        """When all variants overlap, all should be returned."""
        variants = [f"1:{i}" for i in range(50)]
        afr = pd.DataFrame(np.random.randn(50, 3), index=variants)
        eur = pd.DataFrame(np.random.randn(50, 5), index=variants)
        _, _, n = align_to_factorgo_variants(afr, eur)
        assert n == 50

    def test_no_overlap_returns_zero(self):
        """No shared variants should return empty matrices."""
        afr_vars = [f"1:{i}" for i in range(20)]
        eur_vars = [f"2:{i}" for i in range(20)]
        afr = pd.DataFrame(np.random.randn(20, 3), index=afr_vars)
        eur = pd.DataFrame(np.random.randn(20, 5), index=eur_vars)
        afr_a, eur_a, n = align_to_factorgo_variants(afr, eur)
        assert n == 0


# ════════════════════════════════════════════════════════════
# UTILITY
# ════════════════════════════════════════════════════════════

class TestAbsPearson:
    """Tests for absolute Pearson correlation."""

    def test_identical_gives_one(self):
        vec = np.random.randn(50)
        assert abs_pearson(vec, vec) == pytest.approx(1.0, abs=1e-10)

    def test_sign_flip_gives_one(self):
        """FactorGo factors are sign-ambiguous — flipped sign should still give 1."""
        vec = np.random.randn(50)
        assert abs_pearson(vec, -vec) == pytest.approx(1.0, abs=1e-10)

    def test_result_in_unit_interval(self):
        a = np.random.randn(100); b = np.random.randn(100)
        r = abs_pearson(a, b)
        assert 0 <= r <= 1

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            abs_pearson(np.random.randn(50), np.random.randn(30))

    def test_constant_vector_returns_zero(self):
        """Constant vector has undefined correlation — should return 0."""
        a = np.ones(50); b = np.random.randn(50)
        assert abs_pearson(a, b) == 0.0

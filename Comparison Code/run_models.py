"""
run_models.py
-------------
Fits tSVD and NMF on the preprocessed AFR Z-score matrix.

What this script does:
  1. Loads the standardized AFR Z-score matrix from Data/processed/
  2. Fits tSVD (truncated SVD) — the paper's own model-free baseline
  3. Fits NMF (non-negative matrix factorization) — additive baseline
  4. Reports reconstruction quality for both models
  5. Saves the factor loadings to Results/tables/

Why tSVD and NMF on AFR data:
  We are NOT reimplementing FactorGo. We already have the published
  European FactorGo factors from Zenodo. The point of running tSVD
  and NMF here is to get within-ancestry baselines — what does the
  AFR data look like when you decompose it on its own terms? Then
  the gap between FactorGo-EUR and tSVD-AFR tells us how much the
  European training context hurts performance on African data.

tSVD vs NMF (Week 3 course connection):
  tSVD decomposes Z directly as U*S*V^T — allows positive and negative
  loadings, no constraint on structure. This is the paper's own baseline.
  NMF constrains all loadings to be non-negative, forcing parts-based
  additive decomposition. This probes directionality of pleiotropic
  effects — something FactorGo cannot do due to sign ambiguity.

Run from project root:
  python Code/run_models.py

Inputs (from Data/processed/):
  afr_zscore_standardized.tsv.gz

Outputs (to Results/tables/):
  tsvd_afr_loadings.tsv.gz
  tsvd_afr_scores.tsv.gz
  nmf_afr_loadings.tsv.gz
  nmf_afr_scores.tsv.gz
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from utils import run_tsvd, run_nmf

# ── Configuration ─────────────────────────────────────────────────────────────
PROCESSED_DIR = "Data/processed"
RESULTS_DIR   = "Results/tables"

# Number of components — set to 6 to match the number of traits.
# This is the maximum meaningful rank for a 6-column matrix.
# More components than traits adds no information.
N_COMPONENTS = 6

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_processed_matrix(filename: str) -> pd.DataFrame:
    """
    Load a processed matrix from Data/processed/.

    Parameters
    ----------
    filename : str
        Filename inside Data/processed/ (e.g. 'afr_zscore_standardized.tsv.gz').

    Returns
    -------
    pd.DataFrame
        Loaded matrix with variant IDs as index.

    Raises
    ------
    FileNotFoundError
        If the file does not exist — tells the user to run preprocess.py first.
    """
    path = os.path.join(PROCESSED_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Processed file not found: {path}\n"
            f"Run preprocess.py first:\n"
            f"  python Code/preprocess.py"
        )
    print(f"  Loading {filename}...")
    df = pd.read_csv(path, sep="\t", index_col=0, compression="gzip")
    print(f"  Shape: {df.shape[0]:,} variants x {df.shape[1]} columns")
    return df


def save_result(df: pd.DataFrame, filename: str):
    """
    Save a results DataFrame to Results/tables/.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save.
    filename : str
        Output filename.
    """
    path = os.path.join(RESULTS_DIR, filename)
    df.to_csv(path, sep="\t", compression="gzip")
    mb = os.path.getsize(path) / 1e6
    print(f"  Saved {filename} ({mb:.1f} MB)")


def compute_reconstruction_r2(z_matrix: np.ndarray,
                               loadings: np.ndarray,
                               scores: np.ndarray,
                               shifted: bool = False) -> float:
    """
    Compute in-sample reconstruction R² for a factor model.

    R² = 1 - ||Z - L*S^T||^2 / ||Z||^2

    This is the same formula as linear regression R² from Week 2,
    applied to matrix reconstruction rather than a fitted line.

    Parameters
    ----------
    z_matrix : np.ndarray
        Original standardized Z-score matrix.
    loadings : np.ndarray
        Factor loading matrix W. Shape (n_variants, k).
    scores : np.ndarray
        Factor score matrix H^T. Shape (n_traits, k).
    shifted : bool
        If True, compare reconstruction against the shifted (non-negative)
        version of the matrix. Used for NMF since it operates on the
        shifted matrix, not the original.

    Returns
    -------
    float
        Reconstruction R² in [0, 1].
    """
    if shifted:
        # NMF was fit on z_shifted = z - z.min()
        # so we compare reconstruction against that shifted version
        z_compare = z_matrix - z_matrix.min()
    else:
        z_compare = z_matrix

    z_hat       = loadings @ scores.T
    ss_residual = np.sum((z_compare - z_hat) ** 2)
    ss_total    = np.sum(z_compare ** 2)
    return float(np.clip(1 - ss_residual / (ss_total + 1e-10), 0, 1))


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Load standardized AFR matrix ──────────────────────────────────────────
    print("=" * 60)
    print("Loading preprocessed AFR data")
    print("=" * 60)

    afr_std = load_processed_matrix("afr_zscore_standardized.tsv.gz")

    z_mat       = afr_std.values
    variant_ids = list(afr_std.index)
    trait_ids   = list(afr_std.columns)

    print(f"\nMatrix shape: {z_mat.shape[0]:,} variants x {z_mat.shape[1]} traits")
    print(f"Components to extract: {N_COMPONENTS}")

    # ── Fit tSVD ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Fitting tSVD on AFR data")
    print("=" * 60)
    print("tSVD decomposes Z directly as U*S*V^T without forming a")
    print("covariance matrix. This is NOT the same as PCA — tSVD operates")
    print("on the full variant x trait matrix simultaneously, while PCA")
    print("first computes the trait x trait covariance matrix.")
    print()
    print("Key limitation vs FactorGo: tSVD treats all Z-scores equally")
    print("regardless of sample size. FactorGo weights by sqrt(N_i).")
    print("With AFR N ~6,500 vs EUR N ~420,531, this matters a lot.")

    tsvd_loadings, tsvd_scores, tsvd_var = run_tsvd(
        z_mat, variant_ids, trait_ids, N_COMPONENTS
    )

    tsvd_recon_r2 = compute_reconstruction_r2(
        z_mat, tsvd_loadings.values, tsvd_scores.values
    )

    print(f"\ntSVD complete.")
    print(f"  Variance explained per component:")
    for i, v in enumerate(tsvd_var):
        bar = "█" * int(v * 80)
        print(f"    SVD_{i+1}: {v:.4f}  {bar}")
    print(f"  Total variance explained:  {tsvd_var.sum()*100:.1f}%")
    print(f"  Reconstruction R²:         {tsvd_recon_r2:.4f}")

    save_result(tsvd_loadings, "tsvd_afr_loadings.tsv.gz")
    save_result(tsvd_scores,   "tsvd_afr_scores.tsv.gz")

    # ── Fit NMF ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Fitting NMF on AFR data")
    print("=" * 60)
    print("NMF constrains all loadings to be non-negative (W >= 0, H >= 0).")
    print("Forces parts-based decomposition — each factor can only ADD")
    print("to a variant's contribution, never subtract.")
    print()
    print("Why this matters: FactorGo gives |loading| — magnitude only,")
    print("direction unknown due to sign ambiguity. NMF forces positive")
    print("contributions, so when a variant splits across NMF factors but")
    print("stays in one FactorGo factor, it signals opposing directional")
    print("effects that FactorGo cannot distinguish.")
    print()
    print("Input: Z-score matrix shifted to non-negative range (Z - Z.min())")
    print("This preserves relative differences while satisfying NMF constraint.")

    nmf_loadings, nmf_scores, nmf_err = run_nmf(
        z_mat, variant_ids, trait_ids, N_COMPONENTS
    )

    nmf_recon_r2 = compute_reconstruction_r2(
        z_mat, nmf_loadings.values, nmf_scores.values, shifted=True
    )

    print(f"\nNMF complete.")
    print(f"  Reconstruction error: {nmf_err:.4f}")
    print(f"  Reconstruction R²:    {nmf_recon_r2:.4f}")
    print(f"  All loadings >= 0:    {(nmf_loadings.values >= 0).all()}")
    print(f"  All scores >= 0:      {(nmf_scores.values >= 0).all()}")

    save_result(nmf_loadings, "nmf_afr_loadings.tsv.gz")
    save_result(nmf_scores,   "nmf_afr_scores.tsv.gz")

    # ── Summary comparison ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Reconstruction R² comparison (in-sample fit on AFR data)")
    print("=" * 60)
    print(f"  tSVD (k={N_COMPONENTS}): {tsvd_recon_r2:.4f}")
    print(f"  NMF  (k={N_COMPONENTS}): {nmf_recon_r2:.4f}")
    print()
    print("tSVD always achieves optimal L2 reconstruction for given k.")
    print("NMF trades reconstruction quality for interpretable additive")
    print("factors — lower R² here is expected and not a problem.")
    print()
    print(f"All outputs saved to {RESULTS_DIR}/")
    print("Next step: python Code/analysis.py")


if __name__ == "__main__":
    main()

"""
preprocess.py
-------------
Loads and preprocesses all data for the cross-ancestry FactorGo project.

What this script does:
  1. Loads the published FactorGo European factor loadings from Zenodo
  2. Builds an rsID <-> chr:pos:ref:alt mapping from the Pan-UKB variant QC file
     (this is the bridge between EUR rsIDs and AFR chr:pos format)
  3. Loads AFR Z-score vectors for all six traits and converts to rsID format
  4. Aligns AFR variants to the exact 51,399 variants FactorGo used
  5. Standardizes each column to zero mean and unit variance
  6. Saves all processed matrices to Data/processed/ for the next scripts

Why this needs to be its own script:
  The rsID mapping step reads ~28 million rows from a 10GB file.
  Running this once and saving the outputs means run_models.py and
  analysis.py can load the processed files in seconds rather than waiting
  10-15 minutes every time.

Run from project root:
  python Code/preprocess.py

Outputs saved to Data/processed/:
  afr_zscore_aligned.tsv.gz       — raw aligned AFR Z-scores
  afr_zscore_standardized.tsv.gz  — standardized version (input to models)
  eur_loadings_aligned.tsv.gz     — EUR loadings restricted to shared variants
  rsid_map.tsv.gz                 — rsID <-> chr:pos:ref:alt mapping table
"""

import os
import sys
import pandas as pd

# Make sure we can import from the same Code/ folder
sys.path.insert(0, os.path.dirname(__file__))

from utils import (
    build_rsid_to_chrpos_map,
    build_afr_zscore_matrix,
    load_factorgo_loadings,
    align_to_factorgo_variants,
    standardize,
)

# ── Configuration ─────────────────────────────────────────────────────────────
# All paths are relative to the project root (BE_175_Final_Project/)
# Run this script from the project root, not from inside Code/

ZENODO_DIR      = "Data/raw/factorgo_european"
VARIANT_QC_PATH = "Data/raw/panukb/full_variant_qc_metrics.txt"
OUTPUT_DIR      = "Data/processed"

TRAIT_FILES = {
    "BMI":                  "Data/raw/panukb/AFR/bmi.tsv.bgz",
    "Standing Height":      "Data/raw/panukb/AFR/height.tsv.bgz",
    "Basal Metabolic Rate": "Data/raw/panukb/AFR/basal_metabolic_rate.tsv.bgz",
    "Whole Body Fat Mass":  "Data/raw/panukb/AFR/whole_body_fat_mass.tsv.bgz",
    "FEV1":                 "Data/raw/panukb/AFR/fev1.tsv.bgz",
    "Triglycerides":        "Data/raw/panukb/AFR/triglycerides.tsv.bgz",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def check_input_files():
    """
    Verify all required input files exist before doing any heavy work.
    Saves a lot of debugging time if something is missing.

    Returns
    -------
    bool
        True if all files are present, False otherwise.
    """
    required = {
        "FactorGo loadings": os.path.join(
            ZENODO_DIR, "results", "FactorGo.Wm.tsv.gz"),
        "GWAS Z-scores":     os.path.join(
            ZENODO_DIR, "data", "GWAS_Zscore.gz"),
        "Variant QC file":   VARIANT_QC_PATH,
    }
    required.update({f"AFR {k}": v for k, v in TRAIT_FILES.items()})

    all_good = True
    for label, path in required.items():
        if os.path.exists(path):
            mb = os.path.getsize(path) / 1e6
            print(f"  OK   {label:<30} ({mb:.0f} MB)")
        else:
            print(f"  MISSING  {label}: {path}")
            all_good = False

    return all_good


def save(df: pd.DataFrame, filename: str):
    """
    Save a DataFrame to the processed output directory as a gzipped TSV.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save.
    filename : str
        Output filename (e.g. 'afr_zscore_aligned.tsv.gz').
    """
    path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(path, sep="\t", compression="gzip")
    mb = os.path.getsize(path) / 1e6
    print(f"  Saved {filename} ({mb:.1f} MB)")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("STEP 0: Checking input files")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not check_input_files():
        print("\nSome files are missing — see above.")
        print("Check README.md for download instructions.")
        sys.exit(1)

    # ── Step 1: Load EUR factor loadings ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 1: Loading FactorGo European factor loadings")
    print("=" * 60)
    print("File: Data/raw/factorgo_european/results/FactorGo.Wm.tsv.gz")
    print("(pure numeric matrix — variant rsIDs recovered from GWAS_Zscore.gz)")

    eur_loadings = load_factorgo_loadings(ZENODO_DIR)

    # ── Step 2: Build rsID mapping ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Building rsID <-> chr:pos:ref:alt mapping")
    print("=" * 60)
    print("This reads ~28M rows from the variant QC file in 500k-row chunks.")
    print("Takes about 10-15 minutes — only needs to run once.")

    # Check if the mapping was already built from a previous run
    rsid_map_path = os.path.join(OUTPUT_DIR, "rsid_map.tsv.gz")
    if os.path.exists(rsid_map_path):
        print(f"  Found existing mapping at {rsid_map_path} — loading it.")
        rsid_map = pd.read_csv(rsid_map_path, sep="\t", compression="gzip")
        print(f"  {len(rsid_map):,} rsID pairs loaded from cache.")
    else:
        rsid_map = build_rsid_to_chrpos_map(VARIANT_QC_PATH)
        save(rsid_map, "rsid_map.tsv.gz")

    # ── Step 3: Load AFR Z-score matrix ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: Loading AFR Z-score matrix")
    print("=" * 60)
    print("Computing Z = beta_AFR / se_AFR for all six traits.")
    print("Filtering low_confidence_AFR variants during loading.")
    print("Converting chr:pos:ref:alt IDs to rsIDs using the mapping.")

    afr_matrix = build_afr_zscore_matrix(TRAIT_FILES, rsid_map=rsid_map)

    print(f"\nAFR matrix: {afr_matrix.shape[0]:,} variants x "
          f"{afr_matrix.shape[1]} traits")
    print(f"Index format: {afr_matrix.index[0]}")

    # ── Step 4: Align to FactorGo variant set ─────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4: Aligning AFR variants to FactorGo's 51,399-variant set")
    print("=" * 60)
    print("Both datasets must use the same genomic positions for a fair")
    print("comparison — using different variants would confound ancestry")
    print("effects with variant-selection artifacts.")

    afr_aligned, eur_aligned, n_shared = align_to_factorgo_variants(
        afr_matrix, eur_loadings
    )

    if n_shared < 5000:
        print(f"\nWARNING: Only {n_shared:,} shared variants.")
        print("This likely means the rsID mapping did not work correctly.")
        print("Check that VARIANT_QC_PATH points to the right file.")
        sys.exit(1)

    print(f"\nFinal shared variants: {n_shared:,}")

    # ── Step 5: Standardize ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5: Standardizing Z-score matrix")
    print("=" * 60)
    print("Each trait column is centered to zero mean and scaled to unit")
    print("variance. Required before tSVD and NMF so all traits contribute")
    print("equally regardless of their raw Z-score scale.")

    afr_std = standardize(afr_aligned)

    print(f"  Max absolute column mean: "
          f"{afr_std.mean().abs().max():.2e} (should be ~0)")
    print(f"  Max deviation of std from 1: "
          f"{(afr_std.std() - 1).abs().max():.2e} (should be ~0)")

    # ── Step 6: Save outputs ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 6: Saving processed matrices")
    print("=" * 60)

    save(afr_aligned, "afr_zscore_aligned.tsv.gz")
    save(afr_std,     "afr_zscore_standardized.tsv.gz")
    save(eur_aligned, "eur_loadings_aligned.tsv.gz")

    print(f"\nAll outputs saved to {OUTPUT_DIR}/")
    print("Next step: python Code/run_models.py")


if __name__ == "__main__":
    main()

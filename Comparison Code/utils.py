"""
utils.py
--------
Core functions for the cross-ancestry FactorGo generalization project.

Scientific question: Do the pleiotropic factors FactorGo learned from
420,531 Europeans explain the genetic signals seen in 6,636 Africans?

This addresses the health disparity concern directly — if European-trained
models do not generalize to African ancestry data, clinical tools built on
their findings will systematically underserve African patients.

Course method connections (BE 175):
  - tSVD          → Week 3: Dimensionality reduction lecture. Direct matrix
                    decomposition without forming a covariance matrix.
                    Distinct from PCA: tSVD decomposes Z directly as U*S*V^T,
                    PCA first computes the trait-trait covariance matrix.
  - NMF           → Week 3: Same lecture + PCA-NNMF.ipynb lab notebook.
                    Parts-based decomposition with non-negativity constraint.
  - R² projection → Week 2: Regression lecture. Fraction of AFR variance
                    explained by projecting onto a factor subspace. Same
                    formula as linear regression R² = 1 - SS_res/SS_tot.
  - Cross-validation → Week 2: "Does my model work?" lecture. 5-fold CV
                    on variants gives honest generalization estimate vs
                    optimistic in-sample R².
  - Regularization → Week 3: FactorGo's ARD prior is a Bayesian regularizer
                    that automatically prunes uninformative factors.

Key data facts (confirmed from actual files):
  - EUR loadings file: results/FactorGo.Wm.tsv.gz (pure numeric, no labels)
  - EUR variant IDs:   rsIDs from data/GWAS_Zscore.gz (e.g. rs6657440)
  - AFR files:         chr, pos, ref, alt columns — NO rsID column
  - ID bridge:         full_variant_qc_metrics.txt maps rsID ↔ chr:pos:ref:alt
  - AFR QC:            filter low_confidence_AFR == True variants
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.model_selection import KFold


# ════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ════════════════════════════════════════════════════════════════════════

def build_rsid_to_chrpos_map(variant_qc_path: str) -> pd.DataFrame:
    """
    Build a mapping table between rsIDs and chr:pos:ref:alt identifiers.

    This is the critical bridge between two different variant ID systems:
    - FactorGo European loadings use rsIDs (e.g. rs6657440)
    - Pan-UKB AFR summary statistics use chr:pos:ref:alt (e.g. 1:11063:T:G)

    Without this mapping we cannot align the two datasets. The Pan-UKB
    full variant QC file contains both formats for every variant.

    Parameters
    ----------
    variant_qc_path : str
        Path to full_variant_qc_metrics.txt (decompressed) or .txt.bgz.

    Returns
    -------
    pd.DataFrame
        Columns: rsid, chr_pos_ref_alt.
        Use this to join EUR rsIDs to AFR chr:pos:ref:alt IDs.
    """
    print("Building rsID ↔ chr:pos:ref:alt mapping from variant QC file...")
    print("(Reading in chunks to avoid loading full 10GB file into memory)")

    # Detect compression
    compression = "gzip" if variant_qc_path.endswith(".bgz") or \
                            variant_qc_path.endswith(".gz") else None

    # We only need four columns — read in chunks for memory efficiency
    # This is standard practice for large genomics files
    needed_cols = None
    chunks = []
    chunk_size = 500_000

    for i, chunk in enumerate(pd.read_csv(
        variant_qc_path, sep="\t",
        compression=compression,
        chunksize=chunk_size,
        low_memory=False
    )):
        # On first chunk, detect column names
        if i == 0:
            print(f"  Variant QC columns: {chunk.columns.tolist()[:8]}...")
            # Find the rsID column — may be named 'rsid', 'SNP', 'ID'
            rsid_col = next(
                (c for c in chunk.columns
                 if c.lower() in ["rsid", "snp", "id", "variant_id"]),
                None
            )
            chr_col = next(
                (c for c in chunk.columns
                 if c.lower() in ["chrom", "chr", "#chrom"]),
                None
            )
            pos_col = next(
                (c for c in chunk.columns
                 if c.lower() in ["pos", "position", "bp"]),
                None
            )
            ref_col = next(
                (c for c in chunk.columns
                 if c.lower() in ["ref", "a1", "allele1"]),
                None
            )
            alt_col = next(
                (c for c in chunk.columns
                 if c.lower() in ["alt", "a2", "allele2"]),
                None
            )

            if not all([rsid_col, chr_col, pos_col, ref_col, alt_col]):
                raise ValueError(
                    f"Could not find required columns in variant QC file.\n"
                    f"Available columns: {chunk.columns.tolist()}\n"
                    f"Found: rsid={rsid_col}, chr={chr_col}, "
                    f"pos={pos_col}, ref={ref_col}, alt={alt_col}"
                )
            print(f"  Using columns: rsid={rsid_col}, chr={chr_col}, "
                  f"pos={pos_col}, ref={ref_col}, alt={alt_col}")

        # Build chr:pos:ref:alt identifier
        chunk["chr_pos_ref_alt"] = (
            chunk[chr_col].astype(str) + ":" +
            chunk[pos_col].astype(str) + ":" +
            chunk[ref_col].astype(str) + ":" +
            chunk[alt_col].astype(str)
        )

        # Keep only rows with a valid rsID (not '.' or NaN)
        valid = chunk[rsid_col].notna() & (chunk[rsid_col] != ".")
        chunks.append(
            chunk.loc[valid, [rsid_col, "chr_pos_ref_alt"]]
            .rename(columns={rsid_col: "rsid"})
        )

        if (i + 1) % 10 == 0:
            print(f"  Processed {(i+1) * chunk_size:,} rows...")

    mapping = pd.concat(chunks, ignore_index=True).drop_duplicates("rsid")
    print(f"  Mapping built: {len(mapping):,} rsID ↔ chr:pos:ref:alt pairs")
    return mapping


def load_afr_zscore(filepath: str,
                    rsid_map: pd.DataFrame = None) -> pd.Series:
    """
    Load a Pan-UKB per-phenotype summary statistics file and compute
    AFR Z-scores from beta_AFR and se_AFR columns.

    Z-score = beta_AFR / se_AFR

    This is the fundamental unit of GWAS association evidence — a
    standardized effect size that FactorGo, tSVD, and NMF all operate on.
    Z-scores are comparable across studies because they account for
    different sample sizes and trait scales.

    Quality control applied here:
    - Remove variants flagged as low_confidence_AFR (Pan-UKB QC flag)
    - Remove variants with missing beta or se values
    - Remove variants with se = 0 (undefined Z-score)
    - Remove infinite Z-scores

    If rsid_map is provided, the index is converted from chr:pos:ref:alt
    to rsID format so it can be aligned with FactorGo European loadings.

    Parameters
    ----------
    filepath : str
        Path to .tsv.bgz file from Pan-UKB downloads.
    rsid_map : pd.DataFrame, optional
        Mapping from chr_pos_ref_alt → rsid.
        If provided, reindexes output to rsID format.
        Required for alignment with FactorGo EUR loadings.

    Returns
    -------
    pd.Series
        Z-scores indexed by rsID (if rsid_map provided) or chr:pos:ref:alt.
        NaN values removed.

    Raises
    ------
    ValueError
        If required columns are missing from the file.
    """
    # low_memory=False prevents DtypeWarning from mixed chr column types
    # (chr column contains integers 1,2 AND strings like X, MT, Y)
    df = pd.read_csv(filepath, sep="\t", compression="gzip", low_memory=False)

    # Confirm required columns exist
    for col in ["chr", "pos", "ref", "alt", "beta_AFR", "se_AFR"]:
        if col not in df.columns:
            raise ValueError(
                f"Required column '{col}' not found.\n"
                f"Available columns: {df.columns.tolist()}"
            )

    # Apply Pan-UKB low_confidence filter BEFORE building IDs
    # Doing this first reduces memory and speeds up ID construction
    if "low_confidence_AFR" in df.columns:
        n_before = len(df)
        df = df[df["low_confidence_AFR"] != True]
        n_flagged = n_before - len(df)
        if n_flagged > 0:
            print(f"    Removed {n_flagged:,} low-confidence AFR variants")

    # Build chr:pos:ref:alt variant ID
    # Strip whitespace to avoid invisible character mismatches
    # Cast chr to str — handles both int (1) and string (X, MT) chromosomes
    df["chr_pos_ref_alt"] = (
        df["chr"].astype(str).str.strip() + ":" +
        df["pos"].astype(str).str.strip() + ":" +
        df["ref"].astype(str).str.strip() + ":" +
        df["alt"].astype(str).str.strip()
    )
    df = df.set_index("chr_pos_ref_alt")

    # Compute Z-score = effect estimate / standard error
    se = df["se_AFR"].replace(0, np.nan)
    zscore = (df["beta_AFR"] / se).replace([np.inf, -np.inf], np.nan)
    zscore = zscore.dropna()

    # Convert index to rsID format if mapping provided
    if rsid_map is not None:
        lookup       = rsid_map.set_index("chr_pos_ref_alt")["rsid"]
        n_before_map = len(zscore)
        n_overlap    = zscore.index.isin(lookup.index).sum()

        if n_overlap == 0:
            afr_ex = zscore.index[:3].tolist()
            map_ex = lookup.index[:3].tolist()
            raise ValueError(
                f"Zero overlap between AFR IDs and rsID mapping.\n"
                f"AFR examples:     {afr_ex}\n"
                f"Mapping examples: {map_ex}\n"
                f"Formats must match exactly."
            )

        new_index    = zscore.index.map(lookup)
        valid_mask   = new_index.notna()
        zscore       = zscore[valid_mask]
        zscore.index = new_index[valid_mask]
        zscore       = zscore[~zscore.index.duplicated(keep="first")]

        pct = len(zscore) / n_before_map * 100
        print(f"    Mapped to rsIDs: {len(zscore):,} / {n_before_map:,} ({pct:.1f}%)")

    return zscore


def build_afr_zscore_matrix(trait_files: dict,
                             rsid_map: pd.DataFrame = None) -> pd.DataFrame:
    """
    Assemble a variants × traits Z-score matrix from multiple AFR files.

    Loads each trait, aligns on shared variants, and removes any variant
    with missing data in any trait. The resulting matrix is the direct
    input to tSVD, NMF, and the FactorGo projection analysis.

    Course connection: This matrix is the core data object for all three
    models. In the Week 3 dimensionality reduction lecture, the input
    matrix has rows as observations and columns as variables. Here:
    - Rows = genetic variants (51,399 after alignment)
    - Columns = traits (6: BMI, Height, BMR, Fat Mass, FEV1, Triglycerides)

    The matrix is deliberately small (6 traits) because we are not trying
    to run FactorGo on AFR data — we are projecting AFR data onto the
    pre-existing European factor space.

    Parameters
    ----------
    trait_files : dict
        Maps trait display name → file path.
        Example: {"BMI": "Data/raw/panukb/AFR/bmi.tsv.bgz"}
    rsid_map : pd.DataFrame, optional
        rsID ↔ chr:pos:ref:alt mapping for variant ID harmonization.

    Returns
    -------
    pd.DataFrame
        Z-score matrix, shape (n_shared_variants, n_traits).
        No NaN values — only variants with data in all traits retained.
    """
    series = {}
    for name, path in trait_files.items():
        print(f"  Loading {name}...")
        z = load_afr_zscore(path, rsid_map=rsid_map)
        series[name] = z
        print(f"    {len(z):,} variants, index format: {z.index[0]}")

    matrix   = pd.DataFrame(series)
    n_before = len(matrix)
    matrix   = matrix.dropna()
    print(f"\nVariants with complete data across all {len(trait_files)} traits: "
          f"{len(matrix):,} (from {n_before:,} total)")
    return matrix


def load_factorgo_loadings(zenodo_dir: str) -> pd.DataFrame:
    """
    Load published FactorGo European variant loadings from Zenodo archive.

    The loadings file (results/FactorGo.Wm.tsv.gz) is a pure numeric
    matrix with NO row labels and NO column headers. Variant rsIDs are
    recovered from data/GWAS_Zscore.gz which contains them in its first
    column. Column names are assigned as Factor_1 through Factor_100.

    These 100 factors represent shared genetic architecture learned from
    2,483 European traits simultaneously. Each factor is a direction in
    51,399-dimensional variant space that captures a biological theme —
    Factor 1 is body weight/metabolic, Factor 2 is height/musculoskeletal,
    as identified in the original paper.

    Course connection: These factor loading vectors are conceptually
    equivalent to singular vectors from tSVD — directions of maximum
    shared variance across traits — but learned by a probabilistic model
    that explicitly accounts for Z-score measurement uncertainty via
    sqrt(N_i) weighting and uses an ARD prior (Bayesian regularization)
    to prune uninformative factors.

    Parameters
    ----------
    zenodo_dir : str
        Directory containing the unzipped Zenodo archive, specifically
        with subdirectories results/ and data/.

    Returns
    -------
    pd.DataFrame
        Shape (51399, 100).
        Index = rsIDs (e.g. rs6657440).
        Columns = Factor_1 ... Factor_100.

    Raises
    ------
    FileNotFoundError
        If required files are not found. Prints diagnostic information.
    """
    loadings_path = os.path.join(zenodo_dir, "results", "FactorGo.Wm.tsv.gz")
    zscore_path   = os.path.join(zenodo_dir, "data",    "GWAS_Zscore.gz")

    # Validate both files exist before attempting to load
    for path, label in [(loadings_path, "FactorGo.Wm.tsv.gz"),
                         (zscore_path,   "GWAS_Zscore.gz")]:
        if not os.path.exists(path):
            results_contents = []
            for subdir in ["results", "data", ""]:
                d = os.path.join(zenodo_dir, subdir)
                if os.path.isdir(d):
                    results_contents += os.listdir(d)
            raise FileNotFoundError(
                f"Required file '{label}' not found at: {path}\n"
                f"Contents of {zenodo_dir}: {os.listdir(zenodo_dir)}\n"
                f"Contents of subdirectories: {results_contents}\n"
                f"Make sure you ran:\n"
                f"  unzip inner/FactorGo_data_results.zip -d "
                f"{zenodo_dir}"
            )

    # Step 1: Get ordered variant rsIDs from the Z-score file
    # The loadings matrix rows correspond to these variants in order
    print("Step 1: Loading variant rsIDs from GWAS_Zscore.gz...")
    variant_ids = pd.read_csv(
        zscore_path,
        sep="\t",
        usecols=["rsid"],
        compression="gzip"
    )["rsid"].tolist()
    print(f"  {len(variant_ids):,} variant rsIDs loaded")
    print(f"  Example: {variant_ids[0]}, {variant_ids[1]}, {variant_ids[2]}")

    # Step 2: Load the pure numeric loadings matrix
    print("Step 2: Loading FactorGo.Wm.tsv.gz (pure numeric matrix)...")
    loadings_raw = pd.read_csv(
        loadings_path,
        sep="\t",
        header=None,      # No header row in this file
        compression="gzip"
    )
    print(f"  Raw shape: {loadings_raw.shape[0]:,} rows × "
          f"{loadings_raw.shape[1]} columns")

    # Validate dimensions match
    if len(loadings_raw) != len(variant_ids):
        raise ValueError(
            f"Dimension mismatch: loadings has {len(loadings_raw):,} rows "
            f"but variant ID list has {len(variant_ids):,} entries.\n"
            f"The files may be from different versions of the analysis."
        )

    # Step 3: Assign rsID row labels and Factor_k column names
    factor_cols = [f"Factor_{k+1}" for k in range(loadings_raw.shape[1])]
    loadings_df = pd.DataFrame(
        loadings_raw.values,
        index=variant_ids,
        columns=factor_cols
    )

    print(f"  Final shape: {loadings_df.shape[0]:,} variants × "
          f"{loadings_df.shape[1]} factors")
    print(f"  Variant ID format: {loadings_df.index[0]} (rsID)")
    print(f"  Factor names: {loadings_df.columns[:3].tolist()} ...")

    return loadings_df


def load_tsvd_loadings(zenodo_dir: str,
                        variant_ids: list) -> pd.DataFrame:
    """
    Load published tSVD variant loadings from Zenodo archive.

    The paper also published tSVD results (results/tSVD.V.tsv.gz) for
    direct comparison. This function loads them with the same rsID
    labeling as the FactorGo loadings.

    This allows us to compare:
    - FactorGo EUR factors → probabilistic, uncertainty-weighted
    - tSVD EUR factors     → model-free, equal-weight baseline
    Both learned from the same 2,483 European traits.

    Parameters
    ----------
    zenodo_dir : str
        Directory containing unzipped Zenodo archive.
    variant_ids : list
        Ordered rsIDs from GWAS_Zscore.gz (already loaded).

    Returns
    -------
    pd.DataFrame
        Shape (51399, 100). Index = rsIDs, columns = SVD_1...SVD_100.
    """
    tsvd_path = os.path.join(zenodo_dir, "results", "tSVD.V.tsv.gz")

    if not os.path.exists(tsvd_path):
        raise FileNotFoundError(f"tSVD.V.tsv.gz not found at {tsvd_path}")

    print("Loading published tSVD European loadings (tSVD.V.tsv.gz)...")
    tsvd_raw = pd.read_csv(
        tsvd_path, sep="\t", header=None, compression="gzip"
    )

    svd_cols = [f"SVD_{k+1}" for k in range(tsvd_raw.shape[1])]
    tsvd_df  = pd.DataFrame(
        tsvd_raw.values, index=variant_ids, columns=svd_cols
    )

    print(f"  Shape: {tsvd_df.shape[0]:,} variants × {tsvd_df.shape[1]} factors")
    return tsvd_df


# ════════════════════════════════════════════════════════════════════════
# 2. PREPROCESSING
# ════════════════════════════════════════════════════════════════════════

def align_to_factorgo_variants(afr_matrix: pd.DataFrame,
                                eur_loadings: pd.DataFrame) -> tuple:
    """
    Restrict the AFR Z-score matrix to the exact 51,399 variants
    FactorGo used in its European analysis.

    Scientific justification: for the projection R² to measure
    cross-ancestry generalization honestly, we must evaluate on
    exactly the same genomic positions. Using different variants
    would confound ancestry effects with variant-selection artifacts
    — we would not know if low R² means "European factors don't
    generalize" or just "we compared different parts of the genome."

    This is analogous to ensuring train/test sets share the same
    feature space, as covered in the Week 2 validation lecture.

    Parameters
    ----------
    afr_matrix : pd.DataFrame
        AFR Z-score matrix, index should be rsIDs after harmonization.
    eur_loadings : pd.DataFrame
        FactorGo European loadings, index is rsIDs.

    Returns
    -------
    afr_aligned : pd.DataFrame
        AFR matrix restricted to shared variants, in EUR order.
    eur_aligned : pd.DataFrame
        EUR loadings restricted to shared variants.
    n_shared : int
        Number of shared variants.
    """
    shared   = eur_loadings.index.intersection(afr_matrix.index)
    n_shared = len(shared)

    print(f"Variant alignment (both in rsID format):")
    print(f"  FactorGo EUR variants: {len(eur_loadings):,}")
    print(f"  AFR variants:          {len(afr_matrix):,}")
    print(f"  Shared rsIDs:          {n_shared:,} "
          f"({n_shared / len(eur_loadings) * 100:.1f}% of EUR set)")

    if n_shared < 5000:
        print(f"\n  WARNING: Only {n_shared:,} shared variants.")
        print(f"  EUR example: {eur_loadings.index[0]}")
        print(f"  AFR example: {afr_matrix.index[0]}")
        print(f"  Check that rsID mapping was applied correctly.")

    # Return in EUR ordering for consistent matrix operations
    return (afr_matrix.loc[shared].copy(),
            eur_loadings.loc[shared].copy(),
            n_shared)


def standardize(matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize matrix columns to zero mean and unit variance.

    Required before tSVD and NMF so all traits contribute equally
    regardless of raw Z-score scale. Without standardization, traits
    with higher variance would dominate the first decomposition components.

    Course connection: Step 1 of the PCA/tSVD pipeline from the Week 3
    dimensionality reduction lecture. The lecture explicitly notes that
    standardization is necessary when variables are on different scales.
    Here all traits are already in Z-score units, but standardization
    still ensures equal weighting across traits with different signal
    magnitudes.

    Parameters
    ----------
    matrix : pd.DataFrame
        Input matrix (variants × traits). Values are Z-scores.

    Returns
    -------
    pd.DataFrame
        Each column has mean = 0, std = 1.
        Constant columns (std = 0) are set to 0 rather than NaN.
    """
    std = matrix.std()
    std[std == 0] = 1  # Avoid division by zero for constant columns
    result = (matrix - matrix.mean()) / std
    return result.fillna(0)


# ════════════════════════════════════════════════════════════════════════
# 3. DIMENSIONALITY REDUCTION MODELS
#    Course: Week 3 — Dimensionality Reduction lecture + PCA-NNMF.ipynb lab
# ════════════════════════════════════════════════════════════════════════

def run_tsvd(z_matrix: np.ndarray,
             variant_ids: list,
             trait_ids: list,
             n_components: int) -> tuple:
    """
    Run Truncated Singular Value Decomposition (tSVD) on the Z-score matrix.

    tSVD is the paper's own baseline comparison model. It is important
    to be precise about how tSVD differs from PCA:

    PCA computes the covariance matrix C = (1/n) * X^T * X (n_traits × n_traits),
    then finds its eigenvectors. The decomposition operates on the trait-trait
    covariance structure. For our 6-trait matrix this would be a 6×6 matrix.

    tSVD decomposes the raw Z-score matrix Z directly as Z ≈ U * S * V^T
    without ever computing a covariance matrix. It finds the best low-rank
    approximation in the Frobenius norm sense, simultaneously capturing
    structure in both variant space (V) and trait space (U).

    For genomic data at biobank scale — 51,399 variants × 6 traits — tSVD
    is not just faster but mathematically the correct approach because we
    want to decompose the full variant-trait association matrix, not just
    the trait-trait correlations.

    Critical limitation vs FactorGo: tSVD treats all Z-scores as equally
    reliable. A Z-score from an AFR study with N=5,978 gets identical weight
    to one from a EUR study with N=420,531. FactorGo corrects this by
    explicitly scaling each Z-score by sqrt(N_i) before decomposition.
    This distinction is especially important for our analysis because
    AFR sample sizes are ~70x smaller than EUR sample sizes.

    Course connection: Week 3 dimensionality reduction lecture covers both
    PCA and SVD. The lecture shows that for centered data, tSVD and PCA
    give equivalent results, but tSVD is preferred for large sparse matrices.
    The loadings matrix V corresponds to right singular vectors — directions
    of maximum variance in variant space. The scores U*S give trait
    coordinates in the reduced space.

    Parameters
    ----------
    z_matrix : np.ndarray
        Standardized Z-score matrix, shape (n_variants, n_traits).
    variant_ids : list
        Variant rsIDs for output DataFrame index.
    trait_ids : list
        Trait names for output DataFrame index.
    n_components : int
        Number of singular components to extract.

    Returns
    -------
    loadings_df : pd.DataFrame
        Right singular vectors V, shape (n_variants, n_components).
        Each column is a direction of shared variance in variant space.
    scores_df : pd.DataFrame
        Left singular vectors scaled by S: U*S, shape (n_traits, n_components).
        Trait coordinates in the reduced space.
    variance_explained : np.ndarray
        Fraction of total matrix variance per component.
    """
    # sklearn TruncatedSVD expects (n_samples, n_features)
    # We treat traits as samples, variants as features
    # Input: (n_traits, n_variants) = transposed Z-score matrix
    tsvd = TruncatedSVD(n_components=n_components, n_iter=20, random_state=42)
    scores_raw   = tsvd.fit_transform(z_matrix.T)  # (n_traits, k)  = U * S
    loadings_raw = tsvd.components_.T               # (n_variants, k) = V

    cols = [f"SVD_{k+1}" for k in range(n_components)]
    return (
        pd.DataFrame(loadings_raw, index=variant_ids, columns=cols),
        pd.DataFrame(scores_raw,   index=trait_ids,   columns=cols),
        tsvd.explained_variance_ratio_,
    )


def run_nmf(z_matrix: np.ndarray,
            variant_ids: list,
            trait_ids: list,
            n_components: int) -> tuple:
    """
    Run Non-Negative Matrix Factorization on the Z-score matrix.

    NMF decomposes Z ≈ W * H where W ≥ 0 and H ≥ 0. The non-negativity
    constraint forces a parts-based decomposition where each factor
    represents a purely additive biological component with no cancellation
    between factors.

    Biological interpretation: each NMF factor can only ADD to a variant's
    contribution to a trait — it cannot subtract. This maps naturally onto
    how biological pathways work additively. In contrast, tSVD allows
    factors to partially cancel each other, which can produce mathematical
    components that lack biological interpretability.

    Key comparison with FactorGo for this project:
    FactorGo gives |loading| — it identifies which variants matter for
    a factor but cannot tell you the direction of effect (risk-increasing
    or protective) due to sign identifiability ambiguity. NMF forces all
    contributions positive, so when a variant loads highly in FactorGo
    but splits across two NMF components, it indicates that variant has
    opposing directional effects across traits — something FactorGo
    collapses into a single magnitude loading. This is one of the paper's
    explicitly stated limitations that NMF partially addresses.

    Course connection: Covered directly in the Week 3 PCA & NMF lecture
    and the PCA-NNMF.ipynb lab notebook. The course contrasts PCA/tSVD
    (allow negative components, global structure) with NMF (non-negative,
    parts-based, local structure). NMF is presented as the biologically
    motivated choice when factors should represent additive pathways.

    Parameters
    ----------
    z_matrix : np.ndarray
        Standardized Z-score matrix, shape (n_variants, n_traits).
    variant_ids : list
        Variant rsIDs.
    trait_ids : list
        Trait names.
    n_components : int
        Number of NMF components.

    Returns
    -------
    loadings_df : pd.DataFrame
        W matrix, all values ≥ 0. Shape (n_variants, n_components).
    scores_df : pd.DataFrame
        H^T matrix, all values ≥ 0. Shape (n_traits, n_components).
    reconstruction_error : float
        ||Z_shifted - W*H||_F. Lower = better fit.
    """
    # NMF requires non-negative input
    # We shift the matrix so its minimum value is 0
    # This preserves relative differences between Z-scores
    z_shifted = z_matrix - z_matrix.min()

    nmf = NMF(
        n_components=n_components,
        max_iter=500,
        random_state=42,
        init="nndsvda",   # Deterministic non-negative SVD initialization
    )                     # Better convergence than random initialization

    W = nmf.fit_transform(z_shifted)  # (n_variants, k) — variant loadings
    H = nmf.components_.T             # (n_traits, k)   — trait scores

    cols = [f"NMF_{k+1}" for k in range(n_components)]
    return (
        pd.DataFrame(W, index=variant_ids, columns=cols),
        pd.DataFrame(H, index=trait_ids,   columns=cols),
        nmf.reconstruction_err_,
    )


# ════════════════════════════════════════════════════════════════════════
# 4. PROJECTION AND EVALUATION
#    Course: Week 2 — Regression R² and cross-validation
# ════════════════════════════════════════════════════════════════════════

def compute_projection_r2(afr_vec: np.ndarray,
                           factor_loadings: np.ndarray) -> float:
    """
    Compute R² for projecting an AFR trait Z-score vector onto a
    factor loading subspace.

    This is the central evaluation metric of the project. It answers:
    what fraction of the AFR genetic signal variance is captured by
    projecting onto the given factor space?

    Mathematical formulation:
        z_hat = L * (L^T * z)    [orthogonal projection onto column space of L]
        R² = 1 - ||z - z_hat||² / ||z||²

    This is identical in structure to linear regression R² from the
    Week 2 lecture: R² = 1 - SS_residual / SS_total. Here:
    - "model predictions" z_hat = projection of AFR data onto factor space
    - "residuals" (z - z_hat) = AFR signal NOT captured by EUR factors
    - SS_total = total variance in AFR Z-scores

    High R² means the European factor space explains African genetic
    signals well → the model generalizes across ancestries.
    Low R²  means it does not → European-trained tools will perform
    poorly for African patients (the health disparity concern).

    The three factor spaces we evaluate:
    1. FactorGo EUR (100 factors) → generalization test
    2. tSVD AFR (6 factors)       → within-ancestry model-free baseline
    3. NMF AFR (6 factors)        → within-ancestry additive baseline

    Course connection: Week 2 regression lecture defines R² exactly
    this way. The course also shows that R² always improves (or stays
    equal) as you add more dimensions — which is why cross-validation
    is needed to get an honest estimate (see crossval_projection_r2).

    Parameters
    ----------
    afr_vec : np.ndarray
        AFR Z-scores for one trait. Shape (n_shared_variants,).
    factor_loadings : np.ndarray
        Loading matrix defining the projection subspace.
        Shape (n_shared_variants, k).

    Returns
    -------
    float
        R² in [0, 1].
    """
    # Column-normalize loadings for numerically stable orthogonal projection
    norms  = np.linalg.norm(factor_loadings, axis=0, keepdims=True)
    L_norm = factor_loadings / (norms + 1e-10)

    # Orthogonal projection: z_hat = L * L^T * z
    z_hat = L_norm @ (L_norm.T @ afr_vec)

    ss_total    = np.sum(afr_vec ** 2)
    ss_residual = np.sum((afr_vec - z_hat) ** 2)

    return float(np.clip(1 - ss_residual / (ss_total + 1e-10), 0, 1))


def crossval_projection_r2(afr_vec: np.ndarray,
                            factor_loadings: np.ndarray,
                            n_splits: int = 5,
                            random_state: int = 42) -> tuple:
    """
    Estimate projection R² using k-fold cross-validation on variants.

    Variants are split into k folds. For each fold:
    - Training variants (80%): used to define the projection weights
    - Test variants (20%): evaluate R² on held-out genomic regions

    This tests whether the factor space generalizes to unseen parts of
    the genome — a stricter and more honest evaluation than in-sample R².

    Why split on variants and not on individuals?
    We do not have individual-level data — only summary statistics.
    Splitting on variants is the natural generalization test here:
    can the factor structure learned from 80% of genomic regions predict
    associations at the other 20%? This is analogous to asking whether
    a regression model's coefficients generalize beyond the training data.

    Course connection: Directly implements the k-fold cross-validation
    framework from the Week 2 "Does my model work?" lecture. The course
    establishes that in-sample R² is always optimistic — we must hold
    out data to get an honest generalization estimate. The notebook's
    Figure 2 explicitly shows in-sample vs CV R² to demonstrate this.

    Parameters
    ----------
    afr_vec : np.ndarray
        AFR Z-scores for one trait. Shape (n_variants,).
    factor_loadings : np.ndarray
        Loading matrix. Shape (n_variants, k).
    n_splits : int
        Number of folds. Default 5 (standard choice from Week 2 lecture).
    random_state : int
        Seed for reproducible fold assignments.

    Returns
    -------
    mean_r2 : float
        Mean R² across folds — the cross-validated generalization estimate.
    std_r2 : float
        Standard deviation across folds — uncertainty in the estimate.
    fold_r2s : np.ndarray
        Per-fold R² values for plotting fold-level variability.
    """
    kf       = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_r2s = np.zeros(n_splits)

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(afr_vec)):

        # Training fold: defines projection subspace
        L_train = factor_loadings[train_idx]
        z_train = afr_vec[train_idx]
        norms   = np.linalg.norm(L_train, axis=0, keepdims=True)
        L_train_norm = L_train / (norms + 1e-10)

        # Test fold: evaluate on held-out variants
        # Use training norms to normalize test loadings — no data leakage
        L_test      = factor_loadings[test_idx]
        z_test      = afr_vec[test_idx]
        L_test_norm = L_test / (norms + 1e-10)

        # Predict test Z-scores from training projection weights
        z_hat_test  = L_test_norm @ (L_train_norm.T @ z_train)

        ss_total    = np.sum(z_test ** 2)
        ss_residual = np.sum((z_test - z_hat_test) ** 2)
        fold_r2s[fold_idx] = float(
            np.clip(1 - ss_residual / (ss_total + 1e-10), 0, 1)
        )

    return float(fold_r2s.mean()), float(fold_r2s.std()), fold_r2s


def run_full_projection_analysis(afr_aligned: pd.DataFrame,
                                  eur_loadings: np.ndarray,
                                  tsvd_loadings: pd.DataFrame,
                                  nmf_loadings: pd.DataFrame,
                                  trait_domains: dict,
                                  n_cv_splits: int = 5) -> pd.DataFrame:
    """
    Run the complete projection analysis for all traits and all three
    factor spaces, with both in-sample and cross-validated R².

    Three factor spaces compared:
    1. FactorGo EUR — 100 factors learned from 2,483 European traits.
       Tests cross-ancestry generalization. If R² is low, European-trained
       genetic models miss African-specific genetic architecture, directly
       quantifying the health disparity concern.

    2. tSVD AFR — 6 factors from direct matrix decomposition on AFR data.
       Model-free baseline. Treats all Z-scores equally regardless of
       sample size. This is the paper's own comparison model, now applied
       to African data to show what it finds within-ancestry.

    3. NMF AFR — 6 factors with non-negativity constraint on AFR data.
       Additive baseline. Forces parts-based decomposition. Comparison
       with FactorGo reveals variants with opposing directional effects
       that FactorGo's sign ambiguity cannot distinguish.

    The gap between FactorGo-EUR R² and tSVD-AFR R² is the key quantity:
    it measures how much the European training context hurts performance
    on African data compared to a model trained directly on AFR data.

    Parameters
    ----------
    afr_aligned : pd.DataFrame
        AFR Z-score matrix aligned to FactorGo variant set.
    eur_loadings : np.ndarray
        European FactorGo loading matrix, shape (n_shared, 100).
    tsvd_loadings : pd.DataFrame
        tSVD loadings fit on AFR data.
    nmf_loadings : pd.DataFrame
        NMF loadings fit on AFR data.
    trait_domains : dict
        Maps trait name → biological domain string.
    n_cv_splits : int
        Number of cross-validation folds. Default 5.

    Returns
    -------
    pd.DataFrame
        One row per trait. Columns: Trait, Domain, and R² values
        (in-sample and CV) for FactorGo, tSVD, and NMF.
    """
    shared_variants = afr_aligned.index
    tsvd_mat = tsvd_loadings.loc[shared_variants].values
    nmf_mat  = nmf_loadings.loc[shared_variants].values

    results = []
    for trait in afr_aligned.columns:
        afr_vec = afr_aligned[trait].values
        row = {"Trait": trait, "Domain": trait_domains.get(trait, "Unknown")}

        # 1. FactorGo EUR — cross-ancestry generalization
        row["FactorGo_R2_insample"] = compute_projection_r2(
            afr_vec, eur_loadings)
        mu, sd, _ = crossval_projection_r2(
            afr_vec, eur_loadings, n_cv_splits)
        row["FactorGo_R2_cv"]    = mu
        row["FactorGo_R2_cv_sd"] = sd

        # 2. tSVD AFR — model-free within-ancestry baseline
        row["tSVD_R2_insample"] = compute_projection_r2(afr_vec, tsvd_mat)
        mu, sd, _ = crossval_projection_r2(afr_vec, tsvd_mat, n_cv_splits)
        row["tSVD_R2_cv"]    = mu
        row["tSVD_R2_cv_sd"] = sd

        # 3. NMF AFR — additive within-ancestry baseline
        row["NMF_R2_insample"] = compute_projection_r2(afr_vec, nmf_mat)
        mu, sd, _ = crossval_projection_r2(afr_vec, nmf_mat, n_cv_splits)
        row["NMF_R2_cv"]    = mu
        row["NMF_R2_cv_sd"] = sd

        results.append(row)
        print(f"  {trait:<25} "
              f"FactorGo={row['FactorGo_R2_cv']:.3f} | "
              f"tSVD={row['tSVD_R2_cv']:.3f} | "
              f"NMF={row['NMF_R2_cv']:.3f}")

    return pd.DataFrame(results)


# ════════════════════════════════════════════════════════════════════════
# 5. UTILITIES
# ════════════════════════════════════════════════════════════════════════

def abs_pearson(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Compute absolute Pearson correlation between two vectors.

    Absolute value is used because FactorGo factors are identifiable
    only up to sign flip — a loading vector of +0.5 and -0.5 represent
    mathematically equivalent solutions. Both orientations should count
    as high agreement when comparing factor loading vectors.

    Parameters
    ----------
    vec_a : np.ndarray
        First vector.
    vec_b : np.ndarray
        Second vector, same length as vec_a.

    Returns
    -------
    float
        |Pearson r| in [0, 1].
        Returns 0.0 if either vector is constant (undefined correlation).
    """
    if len(vec_a) != len(vec_b):
        raise ValueError(
            f"Vector length mismatch: {len(vec_a)} vs {len(vec_b)}"
        )
    if np.std(vec_a) == 0 or np.std(vec_b) == 0:
        return 0.0
    r, _ = pearsonr(vec_a, vec_b)
    return float(abs(r))

# BE 175 — AFR Factor Analysis Pipeline
### tSVD · NMF · FactorGo Comparison (14-Phenotype Biomarker Subset)

---

## Overview

This pipeline applies Truncated SVD (tSVD) and Non-negative Matrix Factorization (NMF)
to Pan-UKBB African-ancestry (AFR) GWAS summary statistics, then compares the resulting
latent factors to the EUR FactorGo results from Zhang et al. (2023, AJHG).

The script downloads the 15 highest-Neff biomarker phenotypes from Pan-UKBB, builds
a Z-score matrix using the same 51,399 pruned SNPs as the FactorGo paper, and outputs
factor loadings, a comparison table, and a summary figure.

**Runtime:** ~1.5–2 hours (dominated by downloading ~1.5 GB per phenotype file)  
**Internet required:** Yes — files are downloaded live from Pan-UKBB AWS S3

---

## Directory Structure

After unzipping, your folder should look like this:

```
BE_175_Code_14Pheno_tSVD_NMF/
│
├── afr_nmf_pipeline.py              ← main script (run this)
│
├── data copy/                       ← reference files (provided)
│   ├── trait_manifest_TableS6.xlsx
│   ├── allvar.pruned.closesttss.hugo
│   ├── FactorGo.Zm.tsv.gz
│   ├── tSVD.U.tsv.gz
│   └── pheno_manifest_full.tsv.bgz  ← YOU MUST ADD THIS (see below)
│
├── results copy/                    ← output folder (created automatically)
│   └── (empty — outputs go here after running)
│
└── afr_downloads/                   ← temporary download cache (auto-created)
    └── (files downloaded here and deleted after parsing)
```

---

## Required File You Must Add

One file is **not included** in this zip and must be downloaded separately before running:

**`pheno_manifest_full.tsv.bgz`** — the Pan-UKBB full phenotype manifest

Download it here:
```
https://pan-ukb-us-east-1.s3.amazonaws.com/sumstats_release/phenotype_manifest.tsv.bgz
```

After downloading, **rename it** to `pheno_manifest_full.tsv.bgz` and place it in the
`data copy/` folder:

```bash
# Example (adjust path to wherever you downloaded it)
mv ~/Downloads/phenotype_manifest.tsv.bgz "data copy/pheno_manifest_full.tsv.bgz"
```

This file provides AFR sample sizes (n_cases_AFR, n_controls_AFR) that are joined to
the trait manifest to select the highest-powered phenotypes and compute N_eff.

---

## Setup

### 1. Python version
Python 3.8 or later is required.

### 2. Install dependencies
```bash
pip install numpy pandas matplotlib scikit-learn scipy requests openpyxl
```

All packages are standard — no FactorGo installation is needed to run this script.
(FactorGo's pre-computed EUR results are loaded from `FactorGo.Zm.tsv.gz`.)

---

## How to Run

From inside the unzipped folder:

```bash
cd BE_175_Code_14Pheno_tSVD_NMF
python afr_nmf_pipeline.py
```

The script prints progress for all 9 steps to stdout. To save a log:

```bash
python afr_nmf_pipeline.py | tee results\ copy/run_log.txt
```

---

## What the Script Does (Step by Step)

| Step | Description |
|------|-------------|
| 1 | Loads `trait_manifest_TableS6.xlsx`, `allvar.pruned.closesttss.hugo`, and the pre-computed EUR FactorGo matrix |
| 2 | Joins AFR sample sizes from `pheno_manifest_full.tsv.bgz`; selects the 15 biomarker phenotypes with the highest AFR N_eff |
| 3 | Downloads each phenotype's Pan-UKBB GWAS file (~1.5 GB each) and extracts `beta_AFR` / `se_AFR` for the 51,399 target SNPs, matching by `(chr, pos, ref, alt)` |
| 4 | Builds the AFR Z-score matrix: `Z[phenotype, snp] = beta_AFR / (se_AFR × sqrt(N_eff_AFR))` |
| 5 | Runs tSVD (`k = min(10, n_phenotypes − 1)`) and saves phenotype loadings |
| 6 | Scales Z-matrix to [0, 1] with MinMaxScaler, then runs NMF (k same as tSVD); saves W loadings |
| 7 | Correlates AFR NMF components against EUR FactorGo factors (Pearson r, phenotype-level) and saves the comparison table |
| 8 | Generates a 3-panel summary figure (variance explained, tSVD phenotype space, NMF reconstruction) |
| 9 | Prints a final summary with all output paths and a methods caveat |

### The Z-score equation
```
Z_std[phenotype, snp] = (beta_AFR / se_AFR) / sqrt(N_eff_AFR)
```
- For **binary traits**: `N_eff = 2 / (1/n_cases_AFR + 1/n_controls_AFR)`
- For **quantitative traits**: `N_eff = n_AFR`

This matches the standardization used in the FactorGo paper (Zhang et al. 2023) but
applied to AFR-ancestry columns instead of EUR.

---

## Output Files

All outputs are written to `results copy/`:

| File | Description |
|------|-------------|
| `afr_subset_selected.csv` | The 15 phenotypes selected, with phenocodes and N_eff values |
| `afr_zscore_matrix.csv.gz` | Full AFR Z-score matrix (15 phenotypes × 51,399 SNPs) |
| `afr_tsvd_U.csv` | tSVD phenotype loadings (U matrix, 15 × k) |
| `afr_nmf_W_loadings.csv` | NMF phenotype loadings (W matrix, 15 × k) |
| `afr_nmf_comparison_table.csv` | Best-matching EUR FactorGo factor per AFR NMF component, with \|r\| |
| `afr_nmf_main_figure.png` | 3-panel summary figure |

---

## Configuration (optional)

To change the run parameters, edit the `CONFIGURATION` block near the top of
`afr_nmf_pipeline.py`:

```python
K_NMF = 10           # number of factors/components for both tSVD and NMF
N_SUBSET = 15        # number of AFR phenotypes to use
TRAIT_TYPE_FILTER = 'biomarkers'   # restrict to this Pan-UKBB trait_type
                                   # set to None to include all trait types
MIN_AFR_N_EFF = 100  # minimum effective sample size to include a phenotype
DELETE_AFTER_EXTRACT = True  # delete raw .bgz files after parsing to save disk space
```

To run on more phenotypes (slower but more complete), increase `N_SUBSET` or set
`TRAIT_TYPE_FILTER = None`.

---

## Troubleshooting

**`FileNotFoundError: data copy/pheno_manifest_full.tsv.bgz`**
→ You haven't added the phenotype manifest yet. See the "Required File" section above.

**`snps_found=0` for a phenotype**
→ That phenotype's file didn't have matching SNPs (e.g., the chr/pos format differs).
The script skips it and continues. As long as most phenotypes succeed, results are valid.

**`[MISSING COLS]` warning**
→ A phenotype file lacks `beta_AFR` or `se_AFR` columns, meaning Pan-UKBB did not
run an AFR GWAS for that trait. The script skips it automatically.

**Download timeout or slow speed**
→ Pan-UKBB files are ~1.5 GB each. On a slow connection, increase the timeout by
changing `timeout=600` in the `requests.get()` call to a higher value (e.g., `1200`).
Files that were already downloaded to `afr_downloads/` are re-used without re-downloading.

**Script crashes partway through**
→ Re-run as-is. Downloaded files cached in `afr_downloads/` are skipped automatically,
so only remaining phenotypes will be downloaded.

---

## Limitations (for your writeup)

This pilot uses **15 phenotypes** (biomarkers only, top AFR N_eff) rather than the
full 2,483 used in the EUR FactorGo paper, due to the practical constraint that each
Pan-UKBB file is ~1.5 GB and contains genome-wide results for all ancestries with no
pre-filtered AFR-only subset available for download.

Consequently:
- `k` is capped at 14 (n_phenotypes − 1), not 100 as in the paper
- The phenotype-level EUR/AFR correlation (Step 7) is based on n = 15, which is too
  small for statistically robust Pearson r — treat those values as exploratory
- NMF's non-negativity constraint requires scaling the Z-matrix to [0, 1], which
  discards the sign of associations; tSVD operates on the signed Z-scores directly

For a complete AFR vs. EUR comparison, the same pipeline structure could be applied
to all 1,493 AFR-eligible phenotypes (N_eff ≥ 100) given additional compute time.

---

## Reference

Zhang, Y. et al. (2023). Characterizing the genetic architecture of complex traits
using the FactorGo model. *American Journal of Human Genetics*, 111(1).  
https://pmc.ncbi.nlm.nih.gov/articles/PMC10645558/

Pan-UKBB data: https://pan.ukbb.broadinstitute.org/downloads/index.html

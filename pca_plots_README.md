# PCA Plots — BioE 175 Final Project

Generates all PCA visualisations for the PCR analysis component.

## Setup

```bash
pip install -r requirements.txt
```

## Required data files

Place these in the same folder as `pca_plots.py` (or use `--data_dir`):

| File | Description |
|------|-------------|
| `tSVD_U_tsv.gz` | tSVD trait scores U (2483 × 100) |
| `tSVD_D_tsv.gz` | tSVD singular values D (100,) |
| `trait_manifest_TableS6.xlsx` | Pan-UKB trait metadata + heritability |

## Run

```bash
# data files in same folder, results go to ./pca_results/
python pca_plots.py

# specify paths explicitly
python pca_plots.py --data_dir /path/to/data --output_dir ./pca_results
```

## Outputs

| File | Description |
|------|-------------|
| `pca_variance_by_pc.png` | Scree chart PC1–PC7 with biological labels |
| `pca_PC1vPC2.png` | PC1 vs PC2 — trait type & heritability |
| `pca_PC1vPC3.png` | PC1 vs PC3 |
| `pca_PC1vPC4.png` | PC1 vs PC4 |
| `pca_PC1vPC5.png` | PC1 vs PC5 |
| `pca_PC2vPC3.png` | PC2 vs PC3 |
| `pca_PC2vPC4.png` | PC2 vs PC4 |
| `pca_PC2vPC5.png` | PC2 vs PC5 |
| `pca_PC3vPC4.png` | PC3 vs PC4 |
| `pca_PC3vPC5.png` | PC3 vs PC5 |
| `pca_PC4vPC5.png` | PC4 vs PC5 |
| `pca_PC1_PC2_PC3_combined.png` | Combined 2×3 grid for report |

## Biological PC interpretation

| PC | Variance | Theme |
|----|----------|-------|
| PC1 | 13.93% | Body Composition & Metabolic Mass |
| PC2 | 8.42% | Height & Lung Function |
| PC3 | 4.16% | Blood Pressure vs. Body Fat |
| PC4 | 3.26% | Bone Mineral Density |
| PC5 | 2.57% | Red Blood Cell Size vs. Count |
| PC6 | 2.45% | Kidney Function (eGFR) |
| PC7 | 2.26% | RBC Haemoglobin Content |

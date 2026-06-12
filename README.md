# Exploring Shared Genetic Architecture Beyond European Genomes

**BioE 175 — Machine Learning for Bioengineering | UCLA | June 2026**

**Authors:** Ethan Li, Matthew Pham, Sean Phu, Karin Fouad Fahim, Kody Seow

---

## Overview

This project tests whether the pleiotropic factor structure that FactorGo learned from ~420,000 European individuals in the Pan-UK Biobank (Pan-UKBB) generalizes to African-ancestry (AFR) populations. Rather than re-running FactorGo, we take the published European loading matrix from Zhang et al. (2023) and project African GWAS z-scores onto it, measuring cross-validated R² as our generalization metric. tSVD and NMF fit directly on AFR data serve as within-ancestry baselines.

**Core finding:** EUR-trained FactorGo captures almost none of the AFR genetic signal (mean CV R² = 0.003), while NMF fit natively on AFR data achieves CV R² ≈ 0.42 — suggesting a fundamental divergence in pleiotropic architecture at the variant level, not just a modeling artifact.

---

## Repository Structure

```
GWAS-Analysis-BioE175/
├── AFR/                  # AFR z-score preprocessing and variant alignment
├── AFR NMF/              # NMF pipeline on 14 AFR biomarker traits (k=10)
├── AFR NMF Results/      # NMF output: component loadings, EUR comparison table
├── Analyzed Data/        # Processed z-score matrices and intermediate outputs
├── Comparison Code/      # FactorGo projection + tSVD/NMF comparison pipeline
├── Comparison Results/   # CV R² results, heatmaps, bar charts, overfitting plots
├── PCA code/             # PCA visualization on published EUR tSVD factor scores
└── PCA Results/          # PCA scatter plots colored by trait type and heritability
```

---

## Methods

| Method | Role |
|--------|------|
| **FactorGo (EUR, published)** | Pre-trained loading matrix from Zhang et al. (2023); used as projection target |
| **tSVD** | Model-free within-ancestry baseline on AFR z-score matrix |
| **NMF** | Additive, non-negative within-ancestry baseline; probes directionality of effects |
| **5-fold CV on variants** | Honest generalization estimate; avoids in-sample R² inflation |
| **PCA on EUR tSVD scores** | Characterizes the EUR factor space structure across 1,988 Pan-UKBB traits |

---

## Data Sources

All data is publicly available. Raw files are not included due to size (~15 GB).

| Dataset | Source |
|---------|--------|
| FactorGo EUR loading matrix (`FactorGo.Wm.tsv.gz`) | [Zenodo 7765048](https://zenodo.org/records/7765048) |
| Pan-UKBB AFR GWAS summary statistics | [Pan-UKBB Downloads](https://pan.ukbb.broadinstitute.org/downloads/index.html) |
| Pan-UKBB trait manifest (TableS6) | [Zenodo 7765048](https://zenodo.org/records/7765048) |

---

## Requirements

```bash
pip install numpy pandas scikit-learn matplotlib scipy openpyxl
```

---

## How to Run

### 1. Preprocess AFR data
Align AFR z-scores to the 51,399 variants in the FactorGo loading matrix:
```bash
# See AFR/ folder
python AFR/preprocess.py
```

### 2. Run model comparison (6 AFR traits)
Projects AFR z-scores onto EUR FactorGo L matrix and fits tSVD/NMF baselines:
```bash
# See Comparison Code/ folder
python "Comparison Code/run_models.py"
python "Comparison Code/analysis.py"
```

### 3. Run AFR-native NMF (14 biomarker traits)
```bash
# See AFR NMF/ folder
python "AFR NMF/afr_nmf_pipeline_v2.py"
```

### 4. Generate PCA plots
PCA on published EUR tSVD factor scores from Zenodo:
```bash
# See PCA code/ folder
python "PCA code/pca_plots.py"
```

---

## Key Results

| Model | Mean 5-Fold CV R² |
|-------|-------------------|
| FactorGo (EUR, projected onto AFR) | 0.003 |
| tSVD (AFR, k=6) | 1.000 (mathematical ceiling) |
| NMF (AFR, k=6) | 0.424 |

In the 14-trait pilot, AFR NMF components correlated with EUR FactorGo factors at mean |r| = 0.80 across trait loadings, with 2 components showing very strong alignment (|r| > 0.93). However, 4 of 10 components fell below |r| = 0.75, suggesting that while high-level trait groupings (lipid, renal, hepatic) transfer across ancestries, the underlying variant-level weights differ substantially.

---

## Reference

Zhang Z, Jung J, Kim A, Suboc N, Gazal S, Mancuso N. A scalable approach to characterize pleiotropy across thousands of human diseases and complex traits using GWAS summary statistics. *Am J Hum Genet.* 2023;110(11):1863–1874. https://doi.org/10.1016/j.ajhg.2023.09.015

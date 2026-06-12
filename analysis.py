"""
analysis.py
-----------
Runs the core cross-ancestry projection analysis and generates all figures.

What this script does:
  1. Loads aligned AFR Z-scores and EUR FactorGo loadings from Data/processed/
  2. Loads tSVD and NMF AFR loadings from Results/tables/
  3. For each trait and each factor space, computes:
       - In-sample projection R²
       - 5-fold cross-validated R² (splits on variants)
  4. Saves a results table to Results/tables/
  5. Generates four figures to Results/figures/

The projection analysis (Week 2 course connection):
  For each AFR trait Z-score vector z, we project onto a factor loading
  matrix L and compute R² = 1 - ||z - L(L^T z)||^2 / ||z||^2.
  This is identical to linear regression R² — fraction of variance
  explained by the model. Here the "model" is the factor subspace.

Cross-validation (Week 2 course connection):
  We split on variants, not individuals — we only have summary stats.
  Train fold (80% of variants) defines projection weights.
  Test fold (20%) evaluates R² on held-out genomic regions.
  This prevents overfitting to specific variant positions.

Run from project root:
  python Code/analysis.py

Inputs:
  Data/processed/afr_zscore_aligned.tsv.gz
  Data/processed/eur_loadings_aligned.tsv.gz
  Results/tables/tsvd_afr_loadings.tsv.gz
  Results/tables/nmf_afr_loadings.tsv.gz

Outputs:
  Results/tables/projection_results.tsv
  Results/figures/fig1_cv_r2_by_trait.png
  Results/figures/fig2_insample_vs_cv.png
  Results/figures/fig3_r2_by_domain.png
  Results/figures/fig4_heatmap.png
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

sys.path.insert(0, os.path.dirname(__file__))

from utils import run_full_projection_analysis

# ── Configuration ─────────────────────────────────────────────────────────────
PROCESSED_DIR = "Data/processed"
TABLES_DIR    = "Results/tables"
FIGURES_DIR   = "Results/figures"

N_CV_FOLDS = 5  # Standard 5-fold CV from Week 2 lecture

TRAIT_DOMAINS = {
    "BMI":                  "Metabolic",
    "Standing Height":      "Musculoskeletal",
    "Basal Metabolic Rate": "Metabolic",
    "Whole Body Fat Mass":  "Body Composition",
    "FEV1":                 "Pulmonary",
    "Triglycerides":        "Metabolic",
}

# Color scheme — UCLA blue/gold plus two accent colors
DOMAIN_COLORS = {
    "Metabolic":        "#2774AE",
    "Musculoskeletal":  "#FFD100",
    "Body Composition": "#3B9E82",
    "Pulmonary":        "#E05C5C",
}
MODEL_COLORS = {
    "FactorGo (EUR)": "#2774AE",
    "tSVD (AFR)":     "#FFD100",
    "NMF (AFR)":      "#3B9E82",
}

plt.rcParams.update({
    "font.size": 11,
    "figure.dpi": 130,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_required(filename: str, directory: str) -> pd.DataFrame:
    """
    Load a required file, raising a clear error if it is missing.

    Parameters
    ----------
    filename : str
        Name of the file to load.
    directory : str
        Directory to look in.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame with variant IDs as index.

    Raises
    ------
    FileNotFoundError
        With a message explaining which script to run first.
    """
    path = os.path.join(directory, filename)
    if not os.path.exists(path):
        if directory == PROCESSED_DIR:
            prev_script = "python Code/preprocess.py"
        else:
            prev_script = "python Code/run_models.py"
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            f"Run this first: {prev_script}"
        )
    print(f"  Loading {filename}...")
    df = pd.read_csv(path, sep="\t", index_col=0, compression="gzip")
    print(f"  Shape: {df.shape[0]:,} x {df.shape[1]}")
    return df


# ── Figure functions ───────────────────────────────────────────────────────────

def plot_cv_r2_by_trait(results_df: pd.DataFrame):
    """
    Figure 1 — grouped bar chart of 5-fold CV R² per trait per model.

    This is the main result figure. Shows how well each factor space
    explains AFR genetic signals for each trait. Error bars show ±1 SD
    across the 5 CV folds.

    Parameters
    ----------
    results_df : pd.DataFrame
        Output of run_full_projection_analysis.
    """
    traits  = results_df["Trait"].tolist()
    n       = len(traits)
    x       = np.arange(n)
    w       = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))

    model_specs = [
        ("FactorGo (EUR)", "FactorGo_R2_cv", "FactorGo_R2_cv_sd", "#2774AE"),
        ("tSVD (AFR)",     "tSVD_R2_cv",     "tSVD_R2_cv_sd",     "#FFD100"),
        ("NMF (AFR)",      "NMF_R2_cv",      "NMF_R2_cv_sd",      "#3B9E82"),
    ]

    for i, (label, col, sd_col, color) in enumerate(model_specs):
        vals = results_df[col].values
        sds  = results_df[sd_col].values
        ax.bar(x + (i - 1) * w, vals, w,
               label=label, color=color,
               edgecolor="black", linewidth=0.6,
               yerr=sds, capsize=3,
               error_kw={"linewidth": 1})

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{t}\n({TRAIT_DOMAINS[t]})" for t in traits],
        fontsize=8.5, rotation=15, ha="right"
    )
    ax.set_ylabel("5-Fold CV R²", fontsize=11)
    ax.set_title(
        "Do European FactorGo Factors Explain African Ancestry Genetic Signals?\n"
        "Cross-Validated R² by Trait and Model",
        fontsize=12
    )

    ymax = results_df[["FactorGo_R2_cv", "tSVD_R2_cv", "NMF_R2_cv"]].max().max()
    ax.set_ylim(0, min(1.0, ymax * 1.3))
    ax.legend(title="Model (trained on)", fontsize=9)
    ax.axhline(0, color="black", linewidth=0.5)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig1_cv_r2_by_trait.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_insample_vs_cv(results_df: pd.DataFrame):
    """
    Figure 2 — in-sample vs cross-validated R² scatter plots.

    Direct application of the Week 2 cross-validation lesson.
    Points below the diagonal indicate overfitting — the factor
    space memorizes specific variant positions rather than capturing
    generalizable biology.

    Parameters
    ----------
    results_df : pd.DataFrame
        Output of run_full_projection_analysis.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

    model_triples = [
        ("FactorGo (EUR)", "FactorGo_R2_insample", "FactorGo_R2_cv", "#2774AE"),
        ("tSVD (AFR)",     "tSVD_R2_insample",     "tSVD_R2_cv",     "#FFD100"),
        ("NMF (AFR)",      "NMF_R2_insample",       "NMF_R2_cv",      "#3B9E82"),
    ]

    for ax, (name, in_col, cv_col, color) in zip(axes, model_triples):
        in_vals = results_df[in_col].values
        cv_vals = results_df[cv_col].values

        ax.scatter(in_vals, cv_vals, s=80, color=color,
                   edgecolors="black", linewidths=0.6, zorder=3)
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="No overfitting")

        for i, trait in enumerate(results_df["Trait"]):
            ax.annotate(
                trait[:8], (in_vals[i], cv_vals[i]),
                fontsize=7, ha="left", va="bottom",
                xytext=(3, 3), textcoords="offset points"
            )

        ax.set_xlabel("In-Sample R²", fontsize=10)
        ax.set_title(name, fontsize=11)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Cross-Validated R²", fontsize=10)
    fig.suptitle(
        "In-Sample vs Cross-Validated R²  (Week 2: overfitting check)\n"
        "Points below diagonal indicate overfitting to specific variant sets",
        fontsize=11
    )
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig2_insample_vs_cv.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_r2_by_domain(results_df: pd.DataFrame):
    """
    Figure 3 — mean CV R² grouped by biological domain.

    Tests the biological hypothesis: conserved pathways (metabolic,
    musculoskeletal) should generalize better across ancestries than
    domain-specific biology (pulmonary). The paper identified Factor 1
    as brain-mediated BMI biology and Factor 2 as musculoskeletal height
    biology — both are expected to be evolutionarily conserved.

    Parameters
    ----------
    results_df : pd.DataFrame
        Output of run_full_projection_analysis.
    """
    domain_results = results_df.groupby("Domain")[
        ["FactorGo_R2_cv", "tSVD_R2_cv", "NMF_R2_cv"]
    ].mean().reset_index()

    fig, ax = plt.subplots(figsize=(7, 4))

    domains = domain_results["Domain"].tolist()
    x = np.arange(len(domains))
    w = 0.25

    for i, (label, col, color) in enumerate([
        ("FactorGo (EUR)", "FactorGo_R2_cv", "#2774AE"),
        ("tSVD (AFR)",     "tSVD_R2_cv",     "#FFD100"),
        ("NMF (AFR)",      "NMF_R2_cv",      "#3B9E82"),
    ]):
        ax.bar(x + (i - 1) * w, domain_results[col], w,
               label=label, color=color,
               edgecolor="black", linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(domains, fontsize=10)
    ax.set_ylabel("Mean CV R²", fontsize=11)
    ax.set_title(
        "Generalization by Biological Domain\n"
        "Do conserved biological pathways transfer across ancestries?",
        fontsize=12
    )

    ymax = domain_results[["FactorGo_R2_cv", "tSVD_R2_cv", "NMF_R2_cv"]].max().max()
    ax.set_ylim(0, min(1.0, ymax * 1.3))
    ax.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig3_r2_by_domain.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_summary_heatmap(results_df: pd.DataFrame):
    """
    Figure 4 — heatmap of CV R² for all traits and all models.

    Presentation-ready summary. Green = high generalization (EUR factors
    explain AFR data well), red = low generalization.

    Parameters
    ----------
    results_df : pd.DataFrame
        Output of run_full_projection_analysis.
    """
    fig, ax = plt.subplots(figsize=(5, 5))

    heatmap_data = results_df.set_index("Trait")[
        ["FactorGo_R2_cv", "tSVD_R2_cv", "NMF_R2_cv"]
    ].rename(columns={
        "FactorGo_R2_cv": "FactorGo\n(EUR)",
        "tSVD_R2_cv":     "tSVD\n(AFR)",
        "NMF_R2_cv":      "NMF\n(AFR)",
    })

    vmax = max(0.3, float(heatmap_data.values.max()))

    sns.heatmap(
        heatmap_data,
        ax=ax,
        cmap="RdYlGn",
        vmin=0, vmax=vmax,
        annot=True, fmt=".3f",
        annot_kws={"size": 10},
        cbar_kws={"label": "5-Fold CV R²", "shrink": 0.7},
        linewidths=0.5,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=10)
    ax.set_yticklabels(
        [f"{t}\n({TRAIT_DOMAINS[t]})" for t in heatmap_data.index],
        rotation=0, fontsize=8
    )
    ax.set_title(
        "Cross-Ancestry Generalization\n"
        "EUR FactorGo vs AFR tSVD vs AFR NMF",
        fontsize=11
    )

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig4_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def print_summary(results_df: pd.DataFrame, n_shared: int):
    """
    Print a human-readable summary of the key findings.

    Parameters
    ----------
    results_df : pd.DataFrame
        Output of run_full_projection_analysis.
    n_shared : int
        Number of shared variants used in the analysis.
    """
    mean_fg   = results_df["FactorGo_R2_cv"].mean()
    mean_tsvd = results_df["tSVD_R2_cv"].mean()
    mean_nmf  = results_df["NMF_R2_cv"].mean()

    print("=" * 60)
    print("KEY RESULTS")
    print("=" * 60)
    print(f"Shared variants:   {n_shared:,}")
    print(f"Traits analyzed:   {len(results_df)}")
    print(f"CV folds:          {N_CV_FOLDS}")
    print()
    print("Mean cross-validated R² across all traits:")
    print(f"  FactorGo EUR → AFR:  {mean_fg:.3f}")
    print(f"  tSVD AFR → AFR:      {mean_tsvd:.3f}")
    print(f"  NMF  AFR → AFR:      {mean_nmf:.3f}")
    print()

    print("FactorGo EUR → AFR generalization by domain:")
    for domain in results_df["Domain"].unique():
        r2  = results_df[results_df["Domain"] == domain]["FactorGo_R2_cv"].mean()
        bar = "█" * int(r2 * 40)
        print(f"  {domain:<22} {r2:.3f}  {bar}")
    print()

    # The key comparison — how much does European training hurt AFR performance?
    generalization_gap = mean_tsvd - mean_fg
    print(f"Generalization gap (tSVD-AFR minus FactorGo-EUR): "
          f"{generalization_gap:+.3f}")

    if generalization_gap > 0.05:
        print("\nFINDING: A model trained directly on AFR data explains")
        print("substantially more AFR variance than the European-trained")
        print("FactorGo model. This quantifies the health disparity cost")
        print("of European-centric genetic model development.")
    elif generalization_gap > 0:
        print("\nFINDING: FactorGo shows partial cross-ancestry generalization.")
        print("The modest gap suggests EUR factors capture shared biology")
        print("but miss some AFR-specific genetic architecture.")
    else:
        print("\nFINDING: FactorGo EUR factors generalize well to AFR data.")
        print("European and African pleiotropic structure is largely shared.")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    os.makedirs(TABLES_DIR,  exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # ── Load all preprocessed inputs ──────────────────────────────────────────
    print("=" * 60)
    print("Loading processed data and model outputs")
    print("=" * 60)

    afr_aligned   = load_required("afr_zscore_aligned.tsv.gz",  PROCESSED_DIR)
    eur_aligned   = load_required("eur_loadings_aligned.tsv.gz", PROCESSED_DIR)
    tsvd_loadings = load_required("tsvd_afr_loadings.tsv.gz",   TABLES_DIR)
    nmf_loadings  = load_required("nmf_afr_loadings.tsv.gz",    TABLES_DIR)

    n_shared = len(afr_aligned)

    # ── Run projection analysis with cross-validation ─────────────────────────
    print("\n" + "=" * 60)
    print(f"Running projection analysis ({N_CV_FOLDS}-fold CV on variants)")
    print("=" * 60)
    print("For each trait and each factor space:")
    print("  - In-sample R²: project full Z-score vector onto factor subspace")
    print("  - CV R²: 5-fold, train on 80% of variants, test on held-out 20%")
    print()
    print("Three factor spaces:")
    print("  FactorGo EUR (100 factors) — cross-ancestry generalization test")
    print("  tSVD AFR (6 factors)       — model-free within-ancestry baseline")
    print("  NMF AFR  (6 factors)       — additive within-ancestry baseline")
    print()

    results_df = run_full_projection_analysis(
        afr_aligned   = afr_aligned,
        eur_loadings  = eur_aligned.values,
        tsvd_loadings = tsvd_loadings,
        nmf_loadings  = nmf_loadings,
        trait_domains = TRAIT_DOMAINS,
        n_cv_splits   = N_CV_FOLDS,
    )

    results_df = results_df.sort_values("FactorGo_R2_cv", ascending=False)

    # Save results table
    results_path = os.path.join(TABLES_DIR, "projection_results.tsv")
    results_df.to_csv(results_path, sep="\t", index=False)
    print(f"\nResults saved: {results_path}")

    # ── Generate figures ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Generating figures")
    print("=" * 60)

    plot_cv_r2_by_trait(results_df)
    plot_insample_vs_cv(results_df)
    plot_r2_by_domain(results_df)
    plot_summary_heatmap(results_df)

    # ── Print summary ──────────────────────────────────────────────────────────
    print()
    print_summary(results_df, n_shared)

    print()
    print("=" * 60)
    print("Analysis complete. All outputs saved to Results/")
    print("=" * 60)


if __name__ == "__main__":
    main()

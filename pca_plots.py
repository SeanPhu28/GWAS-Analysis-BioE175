"""
pca_plots.py
============
BioE 175 Final Project — PCR Analysis
UCLA Bioengineering 175

Generates all PCA visualisations for the final report:

  1. pca_variance_by_pc.png
       Scree chart for PC1–PC7 with biological interpretation labels
       and a cumulative variance line.

  2. pca_PC1vPC2.png  /  pca_PC1vPC3.png  /  ...  (10 files)
       Individual scatter plots for every pairwise combination of PC1–PC5.
       Each file has two panels: coloured by trait type (BIN/QT) and
       coloured by heritability (log10 h²).

  3. pca_PC1_PC2_PC3_combined.png
       Single combined figure showing PC1vPC2, PC1vPC3, PC2vPC3 in a
       2 × 3 grid (top row = trait type, bottom row = heritability) with
       shared legend and shared colorbar.

Required data files (place in the same folder as this script,
or pass --data_dir to point elsewhere):
  - tSVD_U_tsv.gz
  - tSVD_D_tsv.gz
  - trait_manifest_TableS6.xlsx

Usage
-----
    python pca_plots.py
    python pca_plots.py --data_dir /path/to/data --output_dir ./results

Authors: BioE 175 Group — PCR Component
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


# ── Shared constants ──────────────────────────────────────────────────────────

MAIN_TITLE = (
    "Pleiotropic Factor Structure Separates Trait Types and\n"
    "Predicts Heritability Across 1,988 Pan-UKB GWAS Traits"
)

FOOTNOTE = (
    "Source: Zhang et al. (2023) Am J Hum Genet  |  "
    "tSVD factor scores (U × D)  |  "
    "Traits filtered to h² ∈ [1×10⁻⁴, 1.0]"
)

# Biological interpretation of each PC (from top-loading trait analysis)
PC_THEMES = [
    ("PC1", "Body Composition\n& Metabolic Mass", "#2166AC"),
    ("PC2", "Height &\nLung Function",            "#4393C3"),
    ("PC3", "Blood Pressure\nvs. Body Fat",        "#74ADD1"),
    ("PC4", "Bone Mineral\nDensity",               "#ABD9E9"),
    ("PC5", "Red Blood Cell\nSize vs. Count",      "#F46D43"),
    ("PC6", "Kidney Function\n(eGFR)",             "#D73027"),
    ("PC7", "RBC Haemoglobin\nContent",            "#A50026"),
]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(data_dir: Path) -> tuple:
    """
    Load tSVD components and trait manifest.
    Returns (scores, variance_ratios, log10_h2, trait_meta).

    scores : ndarray (n_valid_traits, 10) — top-10 PC scores
    var    : ndarray (10,)               — explained variance ratios
    y      : ndarray (n_valid_traits,)   — log10(h2_observed)
    meta   : DataFrame                   — trait metadata for valid traits
    """
    print("Loading data...")
    U = pd.read_csv(
        data_dir / "tSVD_U_tsv.gz",
        compression="gzip", header=None, sep="\t"
    ).values
    D = pd.read_csv(
        data_dir / "tSVD_D_tsv.gz",
        compression="gzip", header=None
    ).values.ravel()

    manifest = pd.read_excel(data_dir / "trait_manifest_TableS6.xlsx", header=1)
    manifest = manifest.rename(
        columns={"estimates.final.h2_observed": "h2_observed"}
    )

    h2   = manifest["h2_observed"].values.astype(float)
    mask = (~np.isnan(h2)) & (h2 >= 1e-4) & (h2 <= 1.0)

    X_c  = (U * D)[mask]
    X_c -= X_c.mean(axis=0)

    pca    = PCA(n_components=10, random_state=42)
    scores = pca.fit_transform(X_c)
    var    = pca.explained_variance_ratio_

    y    = np.log10(h2[mask])
    meta = manifest[mask].reset_index(drop=True)

    print(f"  Traits loaded : {mask.sum()} / {len(mask)}")
    print(f"  PC1 variance  : {var[0]*100:.2f}%")
    print(f"  PC1–5 total   : {var[:5].sum()*100:.2f}%\n")

    return scores, var, y, meta


# ── Plot 1: Variance scree chart ──────────────────────────────────────────────

def plot_variance_chart(var: np.ndarray, out: Path) -> None:
    """
    Bar chart of individual variance explained for PC1–PC7,
    with a cumulative variance line and biological theme labels.
    """
    print("Generating variance chart...")
    n      = len(PC_THEMES)
    x      = np.arange(n)
    pct    = var[:n] * 100
    cum    = np.cumsum(pct)
    names  = [p[0] for p in PC_THEMES]
    themes = [p[1] for p in PC_THEMES]
    colors = [p[2] for p in PC_THEMES]

    fig, ax = plt.subplots(figsize=(16, 10), facecolor="#F8F9FA")
    ax.set_facecolor("#FFFFFF")

    bars = ax.bar(x, pct, color=colors, width=0.62, zorder=3,
                  edgecolor="white", linewidth=1.5)

    # Cumulative line on twin axis
    ax2 = ax.twinx()
    ax2.plot(x, cum, color="#222222", lw=2.8, marker="o",
             markersize=10, markerfacecolor="white",
             markeredgecolor="#222222", markeredgewidth=2.4, zorder=5)
    ax2.set_ylabel("Cumulative Variance Explained (%)",
                   fontsize=17, color="#222222", labelpad=14)
    ax2.set_ylim(0, 50)
    ax2.tick_params(labelsize=14, colors="#222222")
    ax2.yaxis.set_major_locator(MultipleLocator(10))
    ax2.spines["top"].set_visible(False)

    # Cumulative % labels above dots
    for i, c in enumerate(cum):
        ax2.text(i, c + 1.6, f"{c:.1f}%",
                 ha="center", va="bottom",
                 fontsize=12, color="#222222", fontweight="bold")

    # Individual % above bars
    for bar, v in zip(bars, pct):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.18,
                f"{v:.2f}%",
                ha="center", va="bottom",
                fontsize=14, fontweight="bold", color="#111111")

    # Biological theme inside each bar
    for bar, theme in zip(bars, themes):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() / 2,
                theme,
                ha="center", va="center",
                fontsize=13, fontweight="bold",
                color="black", linespacing=1.6, zorder=6)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=16, fontweight="bold", color="black")
    ax.set_ylabel("Individual Variance Explained (%)", fontsize=17, labelpad=14)
    ax.set_xlabel("Principal Component", fontsize=17, labelpad=14)
    ax.set_ylim(0, 20)
    ax.set_xlim(-0.55, n - 0.45)
    ax.yaxis.set_major_locator(MultipleLocator(2))
    ax.tick_params(axis="y", labelsize=14)
    ax.grid(axis="y", color="#DDDDDD", lw=0.9, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    ax.legend(
        handles=[Line2D([0], [0], color="#222222", lw=2.8, marker="o",
                        markerfacecolor="white", markeredgecolor="#222222",
                        markersize=10, label="Cumulative variance explained")],
        fontsize=13, loc="upper right", framealpha=0.9, edgecolor="#CCCCCC"
    )

    ax.set_title(
        "Variance Explained by Principal Components PC1–PC7\n"
        "PCA of Pleiotropic Factor Scores Across 1,988 Pan-UKB GWAS Traits",
        fontsize=16, fontweight="bold", pad=18
    )
    fig.text(0.5, -0.02, FOOTNOTE,
             ha="center", fontsize=11, color="#888888", style="italic")

    fig.tight_layout()
    fname = out / "pca_variance_by_pc.png"
    fig.savefig(fname, dpi=160, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved {fname.name}")


# ── Plot 2: Individual pairwise PC scatter plots ──────────────────────────────

def plot_individual_pairs(
    scores: np.ndarray,
    var: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    out: Path,
    n_pcs: int = 5,
) -> None:
    """
    One file per pairwise PC combination (10 files for PC1–PC5).
    Each file: left panel = BIN/QT colouring, right = heritability.
    """
    print("Generating individual pairwise plots...")
    is_bin = meta["BIN_QT"].values == "BIN"
    n      = len(y)
    pairs  = list(combinations(range(n_pcs), 2))

    for xi, yi in pairs:
        xv   = scores[:, xi]
        yv   = scores[:, yi]
        xlab = f"PC{xi+1}  ({var[xi]*100:.2f}% variance explained)"
        ylab = f"PC{yi+1}  ({var[yi]*100:.2f}% variance explained)"
        pname = f"PC{xi+1}vPC{yi+1}"

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        fig.patch.set_facecolor("#F8F8F8")

        # Panel 1: trait type
        for flag, color, marker, lbl, zord in [
            (False, "#4393C3", "s", "Quantitative (QT)", 2),
            (True,  "#D6604D", "o", "Binary (BIN)",      3),
        ]:
            m = is_bin == flag
            ax1.scatter(xv[m], yv[m], c=color, marker=marker,
                        s=28, alpha=0.45, edgecolors="none",
                        label=f"{lbl}  (n={m.sum()})", zorder=zord)

        ax1.axhline(0, color="#AAAAAA", lw=0.8, ls="--")
        ax1.axvline(0, color="#AAAAAA", lw=0.8, ls="--")
        ax1.set_xlabel(xlab, fontsize=16, labelpad=10)
        ax1.set_ylabel(ylab, fontsize=16, labelpad=10)
        ax1.set_title("Coloured by Trait Type",
                      fontsize=17, fontweight="bold", pad=12)
        ax1.legend(fontsize=14, framealpha=0.9, markerscale=1.8,
                   loc="upper left", edgecolor="#CCCCCC")
        ax1.set_facecolor("#FFFFFF")
        ax1.spines[["top", "right"]].set_visible(False)
        ax1.tick_params(labelsize=13)
        ax1.text(0.98, 0.02, f"n = {n} traits",
                 transform=ax1.transAxes, ha="right", va="bottom",
                 fontsize=13, color="#666666")

        # Panel 2: heritability
        sc = ax2.scatter(xv, yv, c=y, cmap="RdYlBu_r",
                         s=28, alpha=0.55, edgecolors="none",
                         vmin=y.min(), vmax=y.max(), zorder=2)
        ax2.axhline(0, color="#AAAAAA", lw=0.8, ls="--")
        ax2.axvline(0, color="#AAAAAA", lw=0.8, ls="--")
        ax2.set_xlabel(xlab, fontsize=16, labelpad=10)
        ax2.set_ylabel(ylab, fontsize=16, labelpad=10)
        ax2.set_title("Coloured by Heritability  (log₁₀ h²)",
                      fontsize=17, fontweight="bold", pad=12)
        ax2.set_facecolor("#FFFFFF")
        ax2.spines[["top", "right"]].set_visible(False)
        ax2.tick_params(labelsize=13)
        cbar = fig.colorbar(sc, ax=ax2, pad=0.02, shrink=0.85)
        cbar.set_label("log₁₀(h²)  →  higher = more heritable",
                       fontsize=13, labelpad=10)
        cbar.ax.tick_params(labelsize=12)
        ax2.text(0.98, 0.02, f"n = {n} traits",
                 transform=ax2.transAxes, ha="right", va="bottom",
                 fontsize=13, color="#666666")

        pct_x = var[xi] * 100
        pct_y = var[yi] * 100
        fig.suptitle(MAIN_TITLE, fontsize=17, fontweight="bold", y=1.02)
        fig.text(0.5, 0.97,
                 f"PC{xi+1} ({pct_x:.2f}%) vs PC{yi+1} ({pct_y:.2f}%)  "
                 f"—  Combined: {pct_x+pct_y:.2f}% of total feature variance",
                 ha="center", fontsize=13, color="#444444", style="italic")

        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fname = out / f"pca_{pname}.png"
        fig.savefig(fname, dpi=160, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  Saved {fname.name}")


# ── Plot 3: Combined PC1/PC2/PC3 figure ───────────────────────────────────────

def plot_combined_pc123(
    scores: np.ndarray,
    var: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    out: Path,
) -> None:
    """
    2 × 3 grid combining PC1vPC2, PC1vPC3, PC2vPC3.
    Top row: coloured by trait type.  Bottom row: coloured by heritability.
    Shared legend and shared colorbar to avoid clutter.
    """
    print("Generating combined PC1/PC2/PC3 figure...")
    is_bin = meta["BIN_QT"].values == "BIN"
    n_bin  = is_bin.sum()
    n_qt   = (~is_bin).sum()
    pairs  = [(0, 1), (0, 2), (1, 2)]
    vmin, vmax = y.min(), y.max()

    fig = plt.figure(figsize=(22, 13), facecolor="#F8F9FA")
    gs  = gridspec.GridSpec(
        2, 4, figure=fig,
        width_ratios=[1, 1, 1, 0.05],
        hspace=0.38, wspace=0.32,
        left=0.07, right=0.93, top=0.88, bottom=0.08,
    )
    axes_type = [fig.add_subplot(gs[0, c]) for c in range(3)]
    axes_h2   = [fig.add_subplot(gs[1, c]) for c in range(3)]
    cax       = fig.add_subplot(gs[1, 3])

    for col, (xi, yi) in enumerate(pairs):
        xv   = scores[:, xi]
        yv   = scores[:, yi]
        xlbl = f"PC{xi+1}  ({var[xi]*100:.2f}% var)"
        ylbl = f"PC{yi+1}  ({var[yi]*100:.2f}% var)"
        ctitle = f"PC{xi+1} vs PC{yi+1}"

        for row, ax in enumerate([axes_type[col], axes_h2[col]]):
            ax.set_facecolor("#FFFFFF")
            ax.spines[["top", "right"]].set_visible(False)
            ax.axhline(0, color="#BBBBBB", lw=0.8, ls="--", zorder=1)
            ax.axvline(0, color="#BBBBBB", lw=0.8, ls="--", zorder=1)
            ax.set_xlabel(xlbl, fontsize=13, labelpad=8)
            ax.set_ylabel(ylbl, fontsize=13, labelpad=8)
            ax.tick_params(labelsize=11)
            ax.set_title(ctitle, fontsize=15, fontweight="bold", pad=10)

            if row == 0:
                for flag, color, marker, lbl in [
                    (False, "#4393C3", "s", f"Quantitative  (n={n_qt})"),
                    (True,  "#D6604D", "o", f"Binary  (n={n_bin})"),
                ]:
                    m = is_bin == flag
                    ax.scatter(xv[m], yv[m], c=color, marker=marker,
                               s=22, alpha=0.45, edgecolors="none",
                               label=lbl, zorder=2 + int(flag))
            else:
                sc = ax.scatter(xv, yv, c=y, cmap="RdYlBu_r",
                                s=22, alpha=0.55, edgecolors="none",
                                vmin=vmin, vmax=vmax, zorder=2)

    # Shared colorbar
    cbar = fig.colorbar(sc, cax=cax)
    cbar.set_label("log₁₀(h²)\nhigher = more heritable",
                   fontsize=12, labelpad=10, linespacing=1.6)
    cbar.ax.tick_params(labelsize=11)

    # Shared legend
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#D6604D",
               markersize=11, label=f"Binary (BIN)  n={n_bin}"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#4393C3",
               markersize=11, label=f"Quantitative (QT)  n={n_qt}"),
    ]
    fig.legend(handles=legend_handles, fontsize=13,
               loc="upper right", bbox_to_anchor=(0.92, 0.96),
               framealpha=0.95, edgecolor="#CCCCCC",
               title="Trait Type", title_fontsize=13)

    # Row labels
    fig.text(0.01, 0.72, "Coloured by\nTrait Type",
             va="center", ha="left", fontsize=13, fontweight="bold",
             color="#333333", rotation=90, linespacing=1.5)
    fig.text(0.01, 0.30, "Coloured by\nHeritability",
             va="center", ha="left", fontsize=13, fontweight="bold",
             color="#333333", rotation=90, linespacing=1.5)

    fig.suptitle(
        "Pleiotropic Factor Structure Separates Trait Types and Predicts Heritability\n"
        "Across 1,988 Pan-UKB GWAS Traits — PC1, PC2, PC3 Pairwise Comparisons",
        fontsize=17, fontweight="bold", y=0.97
    )
    fig.text(0.5, 0.02, FOOTNOTE,
             ha="center", fontsize=10, color="#888888", style="italic")

    fname = out / "pca_PC1_PC2_PC3_combined.png"
    fig.savefig(fname, dpi=160, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved {fname.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(data_dir: str, output_dir: str) -> None:
    data_path = Path(data_dir)
    out_path  = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    scores, var, y, meta = load_data(data_path)

    plot_variance_chart(var, out_path)
    plot_individual_pairs(scores, var, y, meta, out_path)
    plot_combined_pc123(scores, var, y, meta, out_path)

    print(f"\nAll plots saved to: {out_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate all PCA plots for BioE 175 PCR analysis"
    )
    parser.add_argument(
        "--data_dir", type=str, default=".",
        help="Folder containing tSVD_U_tsv.gz, tSVD_D_tsv.gz, "
             "trait_manifest_TableS6.xlsx (default: current folder)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./pca_results",
        help="Folder to save all output PNG files (default: ./pca_results)"
    )
    args = parser.parse_args()
    main(args.data_dir, args.output_dir)
